#include "utils/inc/asserter.hpp"

#include "wave/inc/waveconsts.hpp"
#include "serialize/inc/serwaveop.hpp"

// How to remove unnecessary "value0" keys in generated JSON.
// See:
// https://uscilab.github.io/cereal/archive_specialization.html
namespace cereal
{
    //! Saving for std::map<std::string, std::string>
    template <class Archive, class C, class A> inline
    void save( Archive & ar, std::map<std::string, std::string, C, A> const & map )
    {
        for( const auto & i : map )
        ar( cereal::make_nvp( i.first, i.second ) );
    }

    //! Loading for std::map<std::string, std::string>
    template <class Archive, class C, class A> inline
    void load( Archive & ar, std::map<std::string, std::string, C, A> & map )
    {
        map.clear();

        auto hint = map.begin();
        while( true )
        {
            const auto namePtr = ar.getNodeName();

            if( !namePtr )
                break;

            std::string key = namePtr;
            std::string value; ar( value );
            hint = map.emplace_hint( hint, std::move( key ), std::move( value ) );
        }
    }
} // namespace cereal


namespace kcc {
namespace serialize {
#define KCC_ARCHIVE(X) archive(cereal::make_nvp(WaveOpKey::X, KCC_CONCAT(m_,X)))


//===========================================================================

//===========================================================================
template<>
void
SerWaveOp::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive) const
{
    assert(verify());
    KCC_ARCHIVE(WaveOpType);
    KCC_ARCHIVE(WaveOpName);
    KCC_ARCHIVE(LayerName);
    KCC_ARCHIVE(PreviousWaveOps);

    if (m_PreviousSyncs.size() > 0) {
        KCC_ARCHIVE(PreviousSyncs);
    }

    KCC_ARCHIVE(Order);


    if (m_WaveOpType == wave::WaveOpTypeStr::SBAtomLoad ||
        m_WaveOpType == wave::WaveOpTypeStr::SBAtomSave)
    {
       saveSbAtom(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Pool) {
        savePool(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Reciprocal) {
        saveReciprocal(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegLoad) {
        saveRegLoad(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegStore) {
        saveRegStore(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::MatMul) {
        saveMatMul(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Activation) {
        saveActivation(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ClipByValue) {
        saveClipByValue(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorTensor) {
        saveTensorTensor(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorScalar) {
        saveTensorScalar(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorScalarPtr) {
        saveTensorScalarPtr(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ResAdd) {
        saveResAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ScaleAdd) {
        saveScaleAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Barrier) {
        saveBarrier(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Nop) {
        saveNop(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Minimum) {
        saveMinimum(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Maximum) {
        saveMaximum(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Add) {
        saveAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Sub) {
        saveSub(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Multiply) {
        saveMult(archive);
    } else {
        Assert(false, "Serialization: unsupported WaveOp ", m_WaveOpType);
    }
} // SerWaveOp::save


//===========================================================================
void
SerWaveOp::saveSbAtom(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(Engine);
    KCC_ARCHIVE(SbAddress);
    KCC_ARCHIVE(DataType);
    KCC_ARCHIVE(Length);
    KCC_ARCHIVE(OffsetInFile);
    KCC_ARCHIVE(PartitionStepBytes);
    KCC_ARCHIVE(RefFile);
    KCC_ARCHIVE(RefFileFormat);
    KCC_ARCHIVE(RefFileShape);
    if (m_WaveOpType == wave::WaveOpTypeStr::SBAtomLoad) {
        KCC_ARCHIVE(NumPartitions);
        KCC_ARCHIVE(ContainWeights);

        KCC_ARCHIVE(IfmapReplicationNumRows);
        KCC_ARCHIVE(IfmapReplicationResolution);
        KCC_ARCHIVE(IfmapReplicationStepBytes);

        KCC_ARCHIVE(SrcStepElem);
    } else {
        KCC_ARCHIVE(NumPartitions);
        KCC_ARCHIVE(FinalLayerOfmap);
    }
} // SerWaveOp::saveSbAtom

//===========================================================================
void
SerWaveOp::savePool(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZW);
    saveDst(archive, Dims::XYZ);
    KCC_ARCHIVE(DstStartAtMidPart);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(PoolFrequency);
    KCC_ARCHIVE(PoolFunc);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}


//===========================================================================
void
SerWaveOp::saveReciprocal(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}

//===========================================================================
void
SerWaveOp::saveRegLoad(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(ParallelMode);
}

//===========================================================================
void
SerWaveOp::saveRegStore(cereal::JSONOutputArchive& archive) const
{
    saveDst(archive, Dims::XYZ);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(ParallelMode);
}

//===========================================================================
void
SerWaveOp::saveMatMul(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(FmapXNum);
    KCC_ARCHIVE(FmapXStep);
    KCC_ARCHIVE(FmapYNum);
    KCC_ARCHIVE(FmapYStep);
    KCC_ARCHIVE(FmapZNum);
    KCC_ARCHIVE(FmapZStep);
    KCC_ARCHIVE(IfmapsSbAddress);
    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(NumColumnPartitions);
    KCC_ARCHIVE(NumRowPartitions);
    KCC_ARCHIVE(OutDtype);
    // previous waveops
    KCC_ARCHIVE(PsumBankId);
    KCC_ARCHIVE(PsumBankOffset);
    KCC_ARCHIVE(PsumXNum);
    KCC_ARCHIVE(PsumXStep);
    KCC_ARCHIVE(PsumYNum);
    KCC_ARCHIVE(PsumYStep);
    KCC_ARCHIVE(PsumZNum);
    KCC_ARCHIVE(PsumZStep);
    KCC_ARCHIVE(StartTensorCalc);
    KCC_ARCHIVE(StopTensorCalc);
    // waveop name
    // waveop type
    KCC_ARCHIVE(WeightsSbAddress);

    KCC_ARCHIVE(IfmapReplicationNumRows);
    KCC_ARCHIVE(IfmapReplicationResolution);
    KCC_ARCHIVE(IfmapReplicationShiftAmnt);

    KCC_ARCHIVE(IsDynamicWeights);

    if (utils::DataType::qNeedsQuantization(m_InDtype.c_str())) {
        KCC_ARCHIVE(QuantOffsetIfmaps);
        KCC_ARCHIVE(QuantOffsetWeights);
    }
} // SerWaveOp::saveMatMul


//===========================================================================
void
SerWaveOp::saveActivation(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(ActivationFunc);
    KCC_ARCHIVE(BiasAddEn);
    KCC_ARCHIVE(BiasSbAddress);
    KCC_ARCHIVE(BiasStartAtMidPart);
    KCC_ARCHIVE(Scale);

    KCC_ARCHIVE(BiasDtype);
    KCC_ARCHIVE(NumPartitions);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}


//===========================================================================
void
SerWaveOp::saveClipByValue(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(MinValue);
    KCC_ARCHIVE(MaxValue);
}
 
//===========================================================================
void
SerWaveOp::saveTensorTensor(cereal::JSONOutputArchive& archive) const
{
    saveSrcAB(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(Op);
}

//===========================================================================
void
SerWaveOp::saveTensorScalar(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(Op0);
    KCC_ARCHIVE(Op1);
    KCC_ARCHIVE(ImmVal0);
    KCC_ARCHIVE(ImmVal1);
}

//===========================================================================
void
SerWaveOp::saveTensorScalarPtr(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(Op0);
    KCC_ARCHIVE(Op1);
    KCC_ARCHIVE(ImmPtr0);
    KCC_ARCHIVE(ImmPtr1);
}

//===========================================================================
void
SerWaveOp::saveResAdd(cereal::JSONOutputArchive& archive) const
{
    saveSrcAB(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveMaximum(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveMinimum(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveAdd(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveSub(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveMult(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveScaleAdd(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(Add);
    KCC_ARCHIVE(Scale);
}




//===========================================================================
void
SerWaveOp::saveBarrier(cereal::JSONOutputArchive& /*archive*/) const
{
}

//===========================================================================
void
SerWaveOp::saveNop(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(Engine);
}




//===========================================================================
void
SerWaveOp::saveSrc(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(SrcIsPsum);
    if (m_SrcIsPsum) {
        KCC_ARCHIVE(SrcPsumBankId);
        KCC_ARCHIVE(SrcPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcSbAddress);
        KCC_ARCHIVE(SrcStartAtMidPart);
    }
    switch (dims) {
    case Dims::XYZW:
        KCC_ARCHIVE(SrcWNum);
        KCC_ARCHIVE(SrcWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(SrcZNum);
        KCC_ARCHIVE(SrcZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(SrcYNum);
        KCC_ARCHIVE(SrcYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(SrcXNum);
        KCC_ARCHIVE(SrcXStep);
        break;
    default:
        Assert(false, "Dims to save Src are wrong");
    }
}

//===========================================================================
void
SerWaveOp::saveSrcA(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(InADtype);
    KCC_ARCHIVE(SrcAIsPsum);
    if (m_SrcAIsPsum) {
        KCC_ARCHIVE(SrcAPsumBankId);
        KCC_ARCHIVE(SrcAPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcASbAddress);
        KCC_ARCHIVE(SrcAStartAtMidPart);
    }
    switch (dims) {
    case Dims::XYZW:
        KCC_ARCHIVE(SrcAWNum);
        KCC_ARCHIVE(SrcAWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(SrcAZNum);
        KCC_ARCHIVE(SrcAZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(SrcAYNum);
        KCC_ARCHIVE(SrcAYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(SrcAXNum);
        KCC_ARCHIVE(SrcAXStep);
        break;
    default:
        Assert(false, "Dims to save SrcB are wrong");
    }
}

//===========================================================================
void
SerWaveOp::saveSrcB(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(InBDtype);
    KCC_ARCHIVE(SrcBIsPsum);
    if (m_SrcBIsPsum) {
        KCC_ARCHIVE(SrcBPsumBankId);
        KCC_ARCHIVE(SrcBPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcBSbAddress);
        KCC_ARCHIVE(SrcBStartAtMidPart);
    }
    switch (dims) {
    case Dims::XYZW:
        KCC_ARCHIVE(SrcBWNum);
        KCC_ARCHIVE(SrcBWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(SrcBZNum);
        KCC_ARCHIVE(SrcBZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(SrcBYNum);
        KCC_ARCHIVE(SrcBYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(SrcBXNum);
        KCC_ARCHIVE(SrcBXStep);
        break;
    default:
        Assert(false, "Dims to save SrcB are wrong");
    }
}

//===========================================================================
void
SerWaveOp::saveSrcAB(cereal::JSONOutputArchive& archive, Dims dims) const
{
    saveSrcA(archive, dims);
    saveSrcB(archive, dims);
}

//===========================================================================
void
SerWaveOp::saveDst(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(DstIsPsum);
    if (m_DstIsPsum) {
        KCC_ARCHIVE(DstPsumBankId);
        KCC_ARCHIVE(DstPsumBankOffset);
    } else {
        KCC_ARCHIVE(DstSbAddress);
        KCC_ARCHIVE(DstStartAtMidPart);
    }
    switch (dims) {
    //case Dims::XYZW:
    //    KCC_ARCHIVE(DstWNum);
    //    KCC_ARCHIVE(DstWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(DstZNum);
        KCC_ARCHIVE(DstZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(DstYNum);
        KCC_ARCHIVE(DstYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(DstXNum);
        KCC_ARCHIVE(DstXStep);
        break;
    default:
        Assert(false, "Dims to save Dst are wrong");
    }
}


//===========================================================================
void
SerWaveOp::Sync::save(cereal::JSONOutputArchive& archive) const
{
    std::map<std::string, std::string> m;
    if (m_WithEvent) {
        char buf[256];
        m["sync_type"] = "event";
        sprintf(buf, "%d", static_cast<kcc_int32>(m_EventSync.m_SetMode));
        m["set_mode"] = buf;
        sprintf(buf, "%d", static_cast<kcc_int32>(m_EventSync.m_EventId));
        m["event_id"] = buf;
        sprintf(buf, "%d", static_cast<kcc_int32>(m_EventSync.m_WaitMode));
        m["wait_mode"] = buf;
    } else {
        char buf[256];
        m["sync_type"] = "semaphore";
        m["queue"] = m_SemSync.m_QueueName;
        sprintf(buf, "%d", static_cast<kcc_int32>(m_SemSync.m_TrigOrd));
        m["trig_ord"] = buf;
    }
    //archive(cereal::make_nvp("sync", m));
    archive(m);
}



#undef KCC_ARCHIVE
} // namespace serialize
} // namespace kcc


