"""
src/utils/tree_utils.py

계통수(phylogenetic tree)를 Felsenstein pruning에 쓸 수 있는 형태로 로드/변환.

[구조 설명]
  - TreeStruct: 트리를 인접 리스트 + branch length 배열로 표현
  - load_tree_struct_from_newick(): ete3로 Newick 파일을 읽어 TreeStruct 반환
  - 노드 인덱스는 ete3의 preorder traversal 순서로 부여

[MSA species 순서 연결]
  - msa_codes 텐서의 마지막 축(S)은 특정 species 순서를 가짐
  - leaf_order_names 인자로 그 순서를 명시하면, TreeStruct.leaf_order에
    해당 leaf 노드의 인덱스가 같은 순서로 저장됨
  - 따라서 msa_codes[:, :, s] 가 tree.leaf_order[s] 번 노드에 대응됨
"""

from typing import List
import ete3


class TreeStruct:
    """
    Felsenstein pruning을 위한 트리 컨테이너.

    Attributes
    ----------
    parent : list[int]
        parent[i] = i번 노드의 부모 인덱스. root는 -1.
    children : list[list[int]]
        children[i] = i번 노드의 자식 인덱스 목록.
    branch_length : list[float]
        branch_length[i] = 부모 → i번 노드의 branch length. root는 0.0.
    root_index : int
        루트 노드의 인덱스.
    leaf_order : list[int]
        msa_codes 마지막 축과 같은 순서로 나열된 leaf 노드 인덱스.
        즉, msa_codes[:, :, s] 는 leaf_order[s] 번 노드의 관측값.
    n_nodes : int
        전체 노드 수 (internal + leaf).
    n_leaves : int
        leaf 노드 수 (= len(leaf_order) = S).
    postorder : list[int]
        Felsenstein pruning 순서 (leaves → root). _compute_postorder()로 생성.
    """

    def __init__(
        self,
        parent:        List[int],
        children:      List[List[int]],
        branch_length: List[float],
        root_index:    int,
        leaf_order:    List[int],
    ):
        self.parent        = parent
        self.children      = children
        self.branch_length = branch_length
        self.root_index    = root_index
        self.leaf_order    = leaf_order
        self.n_nodes       = len(parent)
        self.n_leaves      = len(leaf_order)
        self.postorder     = self._compute_postorder()

    def _compute_postorder(self) -> List[int]:
        """DFS post-order (leaves 먼저, root 마지막)."""
        order: List[int] = []

        def dfs(u: int) -> None:
            for v in self.children[u]:
                dfs(v)
            order.append(u)

        dfs(self.root_index)
        return order


def load_tree_struct_from_newick(
    newick_path:      str,
    leaf_order_names: List[str],
) -> TreeStruct:
    """
    Newick 파일을 읽어 TreeStruct 반환.

    Parameters
    ----------
    newick_path : str
        Newick 형식 트리 파일 경로.
        (예: 241-mammalian-2020v2.1.nh.txt — Zoonomia consortium 241종 트리)
    leaf_order_names : list[str]
        msa_codes 의 S 축 순서에 맞는 taxon 이름 목록.
        시뮬레이션 데이터의 경우 npz["taxon_names"] 값을 그대로 넘기면 됨.

    Returns
    -------
    TreeStruct

    Raises
    ------
    KeyError
        leaf_order_names 중 트리에 없는 taxon 이름이 있을 때.
    """
    # ete3로 Newick 읽기 (format=1: 내부 노드 이름 포함)
    t = ete3.Tree(newick_path, format=1)

    # preorder로 모든 노드에 정수 인덱스 부여
    nodes     = list(t.traverse("preorder"))
    node_index = {node: i for i, node in enumerate(nodes)}

    n         = len(nodes)
    parent        = [-1]   * n
    children      = [[]   for _ in range(n)]
    branch_length = [0.0] * n

    for node in nodes:
        idx = node_index[node]
        if node.up is not None:                  # root가 아닌 경우
            p_idx = node_index[node.up]
            parent[idx]        = p_idx
            children[p_idx].append(idx)
            branch_length[idx] = float(node.dist)

    # leaf 이름 → 노드 인덱스 매핑
    leaf_name_to_idx = {
        n.name: node_index[n]
        for n in nodes
        if n.is_leaf()
    }

    # msa_codes S 축 순서대로 leaf 인덱스 정렬
    leaf_order = [leaf_name_to_idx[name] for name in leaf_order_names]

    root_index = node_index[t]   # ete3 Tree 객체 자체가 root

    return TreeStruct(parent, children, branch_length, root_index, leaf_order)
