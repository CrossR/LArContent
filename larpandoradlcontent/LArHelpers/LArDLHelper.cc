/**
 *  @file   larpandoradlcontent/LArHelpers/LArDLHelper.cc
 *
 *  @brief  Implementation of the lar deep learning helper helper class.
 *
 *  $Log: $
 */

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"
#include <c10/core/InferenceMode.h>

namespace lar_dl_content
{

using namespace pandora;

StatusCode LArDLHelper::LoadModel(const std::string &filename, LArDLHelper::TorchModel &model)
{
    try
    {
        model = torch::jit::load(filename);
        std::cout << "Loaded the TorchScript model \'" << filename << "\'" << std::endl;

        // Always set the model to evaluation mode.
        // This disables dropout and batch normalization layers, which is important for inference.
        model.eval();
    }
    catch (const std::exception &e)
    {
        std::cout << "Error loading the TorchScript model \'" << filename << "\':\n" << e.what() << std::endl;
        return STATUS_CODE_FAILURE;
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::InitialiseInput(const at::IntArrayRef dimensions, TorchInput &tensor)
{
    tensor = torch::zeros(dimensions);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::InitialiseInput(const at::IntArrayRef dimensions, TorchInput &tensor, const torch::TensorOptions &options)
{
    tensor = torch::zeros(dimensions, options);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::Forward(TorchModel &model, const TorchInputVector &input, TorchOutput &output)
{
    // Disable the auto-grad engine to save memory and computation time.
    // This is active until it goes out of scope.
    torch::InferenceMode guard;

    output = model.forward(input).toTensor();
}

} // namespace lar_dl_content
