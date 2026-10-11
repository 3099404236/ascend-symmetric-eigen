#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv <= 0) num_cores_aiv = 24;

        uint32_t n = 1;
        uint32_t batch_size = 1;

        const gert::StorageShape *shape_A_storage = context->GetInputShape(0);
        if (shape_A_storage != nullptr) {
            const auto &shape_A = shape_A_storage->GetStorageShape();
            size_t dimA = shape_A.GetDimNum();
            if (dimA >= 2) {
                n = static_cast<uint32_t>(shape_A.GetDim(dimA - 1));
                int64_t total = shape_A.GetShapeSize();
                if (total > 0 && n > 0) {
                    batch_size = static_cast<uint32_t>(total / (n * n));
                }
            }
        }
        if (n == 0) n = 1;
        if (batch_size == 0) batch_size = 1;

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const char *uplo_str = (attrs != nullptr) ? attrs->GetStr(0) : "L";
        uint32_t is_upper = (uplo_str != nullptr && (uplo_str[0] == 'U' || uplo_str[0] == 'u')) ? 1 : 0;

        const gert::Tensor *tensor_A = context->GetRequiredInputTensor(0);
        ge::DataType dtype_A = (tensor_A != nullptr) ? tensor_A->GetDataType() : ge::DT_FLOAT;
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);
        ASCENDC_TPL_SEL_PARAM(context, DT_A);

        LinalgEighTilingData *tiling = context->GetTilingData<LinalgEighTilingData>();
        tiling->n = n;
        tiling->batch_size = batch_size;
        tiling->is_upper = is_upper;

        context->SetBlockDim(num_cores_aiv);

        uint32_t n_stride = ((n + 7) / 8) * 8;
        size_t ws_floats = static_cast<size_t>(num_cores_aiv) * 2 * n * n_stride;
        if (ws_floats < 2 * 512 * 512) ws_floats = 2 * 512 * 512;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = ws_floats * sizeof(float);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *shape_A = context->GetInputShape(0);
        gert::Shape *shape_evals = context->GetOutputShape(0);
        gert::Shape *shape_evecs = context->GetOutputShape(1);
        if (shape_A == nullptr || shape_evals == nullptr || shape_evecs == nullptr) {
            return GRAPH_SUCCESS;
        }

        size_t dimA = shape_A->GetDimNum();
        if (dimA < 2) return GRAPH_FAILED;
        int64_t n = shape_A->GetDim(dimA - 1);

        shape_evals->SetDimNum(dimA - 1);
        for (size_t i = 0; i < dimA - 2; ++i) {
            shape_evals->SetDim(i, shape_A->GetDim(i));
        }
        shape_evals->SetDim(dimA - 2, n);

        *shape_evecs = *shape_A;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        context->SetOutputDataType(1, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgEigh : public OpDef {
    public:
        explicit LinalgEigh(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("eigenvalues")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("eigenvectors")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Attr("UPLO").AttrType(OPTIONAL).String("L");
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgEigh);
}  // namespace ops
