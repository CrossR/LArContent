/**
 *  @file   larpandoracontent/LArObjects/LArSlice.h
 *
 *  @brief  Header file for a simple class representing a slice.
 *
 *  $Log: $
 */
#ifndef LAR_SLICE_H
#define LAR_SLICE_H 1

#include <vector>

#include "Objects/CaloHit.h"

namespace lar_content
{

/**
 *  @brief  Slice class
 */
class Slice
{
public:
    pandora::CaloHitList m_caloHitListU;  ///< The u calo hit list
    pandora::CaloHitList m_caloHitListV;  ///< The v calo hit list
    pandora::CaloHitList m_caloHitListW;  ///< The w calo hit list
    pandora::CaloHitList m_caloHitList3D; ///< The 3D calo hit list
    pandora::VertexList  m_vertexList;    ///< The list of vertices associated to this slice
};

typedef std::vector<Slice> SliceList;

} // namespace lar_content

#endif // #ifndef LAR_SLICE_H
