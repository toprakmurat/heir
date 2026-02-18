#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/OptimizeRelinearizationAnalysis/OptimizeRelinearizationAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

#define DEBUG_TYPE "OptimizeRelinearization"

#define GEN_PASS_DEF_OPTIMIZERELINEARIZATION
#include "lib/Transforms/OptimizeRelinearization/OptimizeRelinearization.h.inc"

struct OptimizeRelinearization
    : impl::OptimizeRelinearizationBase<OptimizeRelinearization> {
  using OptimizeRelinearizationBase::OptimizeRelinearizationBase;

  // Process a single block: recursively handle inner loop bodies first,
  // then strip relins, solve ILP, and insert optimal relins for this block.
  LogicalResult processBlock(Block& block, DataFlowSolver* solver) {
    // Step 1: Recursively process inner blocks of loop-like ops.
    // This ensures inner loop bodies are solved before the outer block.
    for (Operation& op : block) {
      if (op.getNumRegions() > 0 && !isa<secret::GenericOp>(&op)) {
        for (Region& region : op.getRegions()) {
          for (Block& innerBlock : region.getBlocks()) {
            if (failed(processBlock(innerBlock, solver))) {
              return failure();
            }
          }
        }
      }
    }

    // Step 2: Strip relins in THIS block only (not walking into nested
    // regions). Inner blocks have already been processed and have their
    // own optimal relins in place.
    for (Operation& op : llvm::make_early_inc_range(block)) {
      if (auto relinOp = dyn_cast<mgmt::RelinearizeOp>(&op)) {
        relinOp.getResult().replaceAllUsesWith(relinOp.getOperand());
        relinOp.erase();
      }
    }

    // Step 3: Solve the ILP for THIS block only.
    OptimizeRelinearizationAnalysis analysis(
        &block, solver, useLocBasedVariableNames, allowMixedDegreeOperands);
    if (failed(analysis.solve())) {
      block.getParentOp()->emitError(
          "Failed to solve the relinearization optimization problem");
      return failure();
    }

    // Step 4: Collect ops that need relins inserted after them.
    OpBuilder b(&getContext());
    SmallVector<Operation*> opsToRelin;
    for (Operation& op : block) {
      if (analysis.shouldInsertRelin(&op)) {
        opsToRelin.push_back(&op);
      }
    }

    // Step 5: Insert relins at the optimal locations.
    for (Operation* op : opsToRelin) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting relin after: " << op->getName() << "\n");
      b.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        auto reduceOp = mgmt::RelinearizeOp::create(b, op->getLoc(), result);
        result.replaceAllUsesExcept(reduceOp.getResult(), {reduceOp});
      }
    }

    return success();
  }

  void processSecretGenericOp(secret::GenericOp genericOp,
                              DataFlowSolver* solver) {
    if (failed(processBlock(*genericOp.getBody(), solver))) {
      return signalPassFailure();
    }
  }

  void runOnOperation() override {
    Operation* module = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    module->walk(
        [&](secret::GenericOp op) { processSecretGenericOp(op, &solver); });

    // optimize-relinearization will invalidate mgmt attr
    // so re-annotate it

    // temporary workaround for B/FV and all schemes of Openfhe
    auto baseLevel = 0;
    if (moduleIsBFV(getOperation()) || moduleIsOpenfhe(getOperation())) {
      // inherit mulDepth information from existing mgmt attr.
      mgmt::MgmtAttr mgmtAttr = nullptr;
      getOperation()->walk([&](secret::GenericOp op) {
        for (auto i = 0; i != op->getBlock()->getNumArguments(); ++i) {
          if ((mgmtAttr = dyn_cast<mgmt::MgmtAttr>(op.getOperandAttr(
                   i, mgmt::MgmtDialect::kArgMgmtAttrName)))) {
            break;
          }
        }
      });

      if (!mgmtAttr) {
        getOperation()->emitError(
            "No mgmt attribute found in the module for B/FV");
        return signalPassFailure();
      }

      baseLevel = mgmtAttr.getLevel();
    }

    OpPassManager pipeline("builtin.module");
    mgmt::AnnotateMgmtOptions annotateMgmtOptions;
    annotateMgmtOptions.baseLevel = baseLevel;
    pipeline.addPass(mgmt::createAnnotateMgmt(annotateMgmtOptions));
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
