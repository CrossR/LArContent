/**
 *  @file   larpandoradlcontent/LArSlicing/KnnKDTree.h
 *
 *  @brief  Header file for a contiguous memory k-NN KD-Tree
 *
 *  $Log: $
 */
#ifndef LAR_KNN_KD_TREE_H
#define LAR_KNN_KD_TREE_H 1

#include <queue>
#include <utility>
#include <vector>

namespace lar_content
{

/**
 * @brief  KnnKdTree class
 */
class KnnKdTree
{
public:
    /**
     * @brief Lightweight struct to hold node information for the KD-Tree.
     *        Should be more cache-friendly, as it is smaller.
     */
    struct KnnNode
    {
        float coords[3];
        int original_id;
    };

    /**
     * @brief  Constructor builds the KD-Tree immediately from the provided nodes
     * * @param  inputNodes The flat list of nodes to build the tree from
     */
    KnnKdTree(const std::vector<KnnNode> &inputNodes);

    /**
     * @brief  Search for the k nearest neighbors of a query point
     *
     * @param  query The node to search around
     * @param  k The number of neighbors to find
     *
     * @return A vector of the original_ids of the nearest neighbors
     */
    std::vector<int> FindNearestNeighbours(const KnnNode &query, const int k) const;

private:
    /**
     * @brief Recursive tree builder
     */
    void BuildTree(int start, int end, int depth);

    /**
     * @brief Recursive tree searcher
     */
    void SearchTree(int start, int end, int depth, const KnnNode &query, const int k, std::priority_queue<std::pair<float, int>> &pq) const;

    std::vector<KnnNode> m_nodes; ///< The contiguous array holding the sorted tree
};

} // namespace lar_content

#endif // LAR_KNN_KD_TREE_H