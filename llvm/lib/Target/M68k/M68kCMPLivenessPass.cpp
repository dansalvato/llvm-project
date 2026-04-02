//===-- M68kCMPLivenessPass.cpp - CMP Liveness pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a pass that runs immediately before register allocation.
/// The pass ensures registers used as operands in compare instructions remain
/// live until the end of the block. This way, post-RA COPY instructions become
/// easier to move to before the compare, since they will not reuse the same
/// register(s) being used for the compare.
///
//===----------------------------------------------------------------------===//

#include "M68k.h"
#include "M68kFrameLowering.h"
#include "M68kInstrInfo.h"
#include "M68kMachineFunction.h"
#include "M68kSubtarget.h"

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "m68k-cmp-liveness"
#define PASS_NAME "M68k CMP liveness pass"

namespace {

class M68kCMPLiveness : public MachineFunctionPass {
public:
  static char ID;

  const M68kSubtarget *STI;
  const M68kInstrInfo *TII;
  const M68kRegisterInfo *TRI;
  const M68kMachineFunctionInfo *MFI;
  const M68kFrameLowering *FL;

  M68kCMPLiveness() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    STI = &MF.getSubtarget<M68kSubtarget>();
    TII = STI->getInstrInfo();
    TRI = STI->getRegisterInfo();
    MFI = MF.getInfo<M68kMachineFunctionInfo>();
    FL = STI->getFrameLowering();

    bool Modified = false;

    for (auto &MBB : MF) {
      auto MI = MBB.rbegin(), E = MBB.rend();

      while (MI != E) {
        if (!MI->hasRegisterImplicitUseOperand(M68k::CCR) || !MI->isBranch()) {
          MI = std::next(MI);
          continue;
        }

        // We found an instruction that uses CCR
        auto User = MI;
        MI = std::next(MI);

        // Find the related instruction that defines CCR
        while (!MI->definesRegister(M68k::CCR, TRI)) {
          MI = std::next(MI);
        }

        // Maybe the CCR is set by an instruction that doesn't use any regs
        auto UsedRegs = MI->all_uses();
        if (UsedRegs.empty()) {
          MI = std::next(MI);
          continue;
        }

        // Now, kill registers on the CCR user instead
        for (auto Op : UsedRegs) {
          auto Reg = Op.getReg();
          if (MI->killsRegister(Reg, TRI)) {
            User->addRegisterKilled(Reg, TRI, true);
            MI->clearRegisterKills(Reg, TRI);
            Modified = true;
          }
        }

        MI = std::next(MI);
      }
    }

    return Modified;
  }
};

char M68kCMPLiveness::ID = 0;
} // anonymous namespace.

INITIALIZE_PASS(M68kCMPLiveness, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createM68kCMPLivenessPass() {
  return new M68kCMPLiveness();
}
