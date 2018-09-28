#pragma once

#ifndef KCC_WAVECODE_WAVECODEWAVEOP_H
#define KCC_WAVECODE_WAVECODEWAVEOP_H

#include <string>
#include <cstdio>
#include <sstream>


#include "aws_tonga_isa_tpb_common.h"

#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisaclear.hpp"
#include "compisa/inc/compisamatmul.hpp"



#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"

#include "wave/inc/waveop.hpp"
#include "wave/inc/waveedge.hpp"

#include "wavecode/inc/wavecode.hpp"

// struct TONGA_ISA_TPB_INST_EVENTS;

namespace kcc {

class WaveCode;

namespace wave {
    class WaveOp;
}

namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to)                        \
    Assert((edge)->gEventId() != events::EventId_Invalid(),     \
    "WaveEdge from waveop ", (from)->gName(), " to waveop ",    \
    (to)->gName(), " has no event")

class WaveCode;


class WaveCodeWaveOp {
protected:
    using WaveCodeRef = WaveCode&;
public:
    //----------------------------------------------------------------
    WaveCodeWaveOp(WaveCodeRef wavecode);

    virtual ~WaveCodeWaveOp()
    {}

    virtual void generate(wave::WaveOp* waveOp) = 0;

    //----------------------------------------------------------------
    wave::WaveOp* gWaveOp() const {
        return m_WaveOp;
    }

    //----------------------------------------------------------------
    void rWaveOp(wave::WaveOp* waveOp) {
        m_WaveOp = waveOp;
    }

protected:
    enum PatDim {
        PatDim_X = 0,
        PatDim_Y = 1,
        PatDim_Z = 2,
        PatDim_W = 3,
    };
    void epilogue(const wave::WaveOp* waveOp);
    bool qParallelStreams() const;

    /* Process incoming edges for instructions without embedded events (no SYNC)
    * 1. Issue WAIT instruction for all in-edges
    */
    kcc_int32 processIncomingEdges(wave::WaveOp* waveop);

    kcc_int32 processIncomingEdges(wave::WaveOp* waveop, EngineId engineId);

    /* Process incoming edges for instructions with embedded events (with SYNC)
    * 1. Assign embedded wait for one in-edge
    * 2. Issue WAIT instruction for other in-edges
    */
    kcc_int32 processIncomingEdges(wave::WaveOp* waveop, TONGA_ISA_TPB_INST_EVENTS& sync);

    /* Process incoming edges for instructions with embedded events (with SYNC)
    * But don't assign embedded events to instruction
    * 1. Remember embedded wait id/mode for one in-edge
    */
    kcc_int32 processIncomingEdges(wave::WaveOp* waveop,
                        events::EventId& waitEventId, events::EventWaitMode& waitEventMode);

    kcc_int32 processIncomingEdgesForceWait(wave::WaveOp* waveop, EngineId engId,
                        events::EventId& waitEventId, events::EventWaitMode& waitEventMode);

    kcc_int32 processIncomingEdges(wave::WaveOp* waveop, EngineId engineId,
                   bool allowEmb,
                   TONGA_ISA_TPB_INST_EVENTS* sync,
                   events::EventId* waitEventId, events::EventWaitMode* waitEventMode);

    void findFirstSetEventIdMode(wave::WaveOp* waveop, events::EventId& setEventId,
                            events::EventSetMode& setEventMode);

    /* Process outgoing edges for instructions without embedded events (no SYNC)
    * 1. Issue SET instruction for all out-edges
    */
    kcc_int32 processOutgoingEdges(wave::WaveOp* waveop);

    /* Process outgoing edges for instructions with embedded events (with SYNC)
    * 1. Assign embedded set for one out-edge
    * 2. Issue SET instruction for other out-edges
    */
    template <typename INST>
    bool processOutgoingEdges(wave::WaveOp* waveop, INST& instr)
    {
        bool instructionWritten = false; // for no succ edges with event, return false
        bool firstEmb = true;
        unsigned int  numSuccEdgesToSync = 0;

        for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
            if (succWaveEdge->qNeedToImplementSync()) {
                numSuccEdgesToSync++;
            }
        }

        for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
            if (! succWaveEdge->qNeedToImplementSync()) {
                continue;
            }
            if (succWaveEdge->qChosenForSuccSbAtom()) {
                continue;
            }

            if (firstEmb) {
                firstEmb = false;
                std::ostringstream oss;
                oss << waveop->gOrder() << "-" <<  waveop->gName();
                if (waveop->gType() == WaveOpType::MatMul && numSuccEdgesToSync > 1) {
                    // kaena-531: There's only 1 delay from MM to following event set instr when there are
                    // multiple SETs (multiple dependencies), so to properly trigger a dependent load,
                    // there must be an event from MM to a WAIT followed by the first SETs (no longer embedded)
                    // followed by the next series of SETs. Reusing the last event ID (255) since that
                    // was used only for the start of inference.
                    // 1. MatMul sets a reserved event
                    instr.inst_events.set_event_idx  = events::EventMgr::EventId_MMStartMultiSet();
                    instr.inst_events.set_event_mode = events::eventSetMode2Isa(succWaveEdge->gSetEventMode());
                    m_WaveCode.SaveName(instr, oss.str().c_str());
                    m_WaveCode.writeInstruction(instr); // this requires template
                    // 2. Wait for reserved event
                    {
                        compisa::WaitInstr waitEventInstr;
                        waitEventInstr.event_idx         = events::EventMgr::EventId_MMStartMultiSet();
                        waitEventInstr.wait_event_mode   = events::eventWaitMode2Isa(events::EventWaitMode::WaitThenClear);
                        m_WaveCode.SaveName(waitEventInstr, oss.str().c_str());
                        m_WaveCode.writeInstruction(waitEventInstr, waveop->gEngineId());
                    }
                    // 3. Set the actual event scheduled
                    {
                        compisa::SetInstr setEventInstr;
                        setEventInstr.event_idx          = succWaveEdge->gEventId();
                        m_WaveCode.SaveName(setEventInstr, oss.str().c_str());
                        m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
                    }
                }
                else {
                    instr.inst_events.set_event_idx    = succWaveEdge->gEventId();
                    instr.inst_events.set_event_mode  = events::eventSetMode2Isa(
                                                    succWaveEdge->gSetEventMode());
                    m_WaveCode.SaveName(instr, oss.str().c_str());
                    m_WaveCode.writeInstruction(instr); // this requires template
                }
                instructionWritten = true;
                //std::cout << waveop->gName() << " (embedded) " << succWaveEdge->gEventId() << std::endl;
            } else {
                std::ostringstream oss;
                oss << waveop->gOrder() << "-" <<  waveop->gName();
                compisa::SetInstr setEventInstr;
                setEventInstr.event_idx = succWaveEdge->gEventId();
                m_WaveCode.SaveName(setEventInstr, oss.str().c_str());
                m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
                //std::cout << waveop->gName() << " (not embedded) " << succWaveEdge->gEventId() << " engine " << utils::engineId2Str(waveop->gEngineId()) << std::endl;
            }
        }
        return instructionWritten;
    }

    //void writeWaitOrWaitClearInstr(const wave::WaveEdge* edge, EngineId engineId);
    //void writeWaitOrWaitClearInstr(events::EventId evntId, events::EventWaitMode waitEventMode,
    //                EngineId engineId, const char* const dbgTxt);


    template<typename MEM_ACCESS>
    static void initMemAccess(MEM_ACCESS& mem_pattern) // e.g., TONGA_ISA_TPB_MEM_ACCESS_3D
    {
        const int numDims = sizeof(mem_pattern.step_elem)/sizeof(mem_pattern.step_elem[0]);
        for (int i = 0; i < numDims ; ++i) {
            AssignWithSizeCheck(mem_pattern.step_elem[i], 0);
            AssignWithSizeCheck(mem_pattern.num_elem[i], 1);
        }
    }

    bool qGenerateKelf() const {
        return m_WaveCode.qGenerateKelf();
    }

    //void rGenerateKelf(bool genKelf) {
    //    m_WaveCode.rGenerateKelf(genKelf);
    //}

private:

protected:
    WaveCodeRef     m_WaveCode;
    wave::WaveOp*   m_WaveOp;
};

}}

#endif // KCC_WAVECODE_WAVECODEWAVEOP_H

