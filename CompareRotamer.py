from Bio.PDB import PDBParser, Superimposer, NeighborSearch, is_aa
import numpy as np
from scipy.spatial.distance import cdist
from Bio.Align import substitution_matrices
from Bio import pairwise2

def calculate_normal_vector(atom1, atom2, atom3):
    """
    计算由三个原子定义的平面的法向量。
    """
    coord1, coord2, coord3 = atom1.get_coord(), atom2.get_coord(), atom3.get_coord()
    v1, v2 = coord2 - coord1, coord3 - coord2  # 定义两个向量
    normal_vector = np.cross(v1, v2)  # 计算叉积得到法向量
    return normal_vector / np.linalg.norm(normal_vector)  # 标准化为单位向量

def get_residue_plane_normal_vector(residue):
    """
    获取残基中 N, CA, C 原子构成的平面的法向量。
    """
    try:
        return calculate_normal_vector(residue['N'], residue['CA'], residue['C'])
    except KeyError:
        return None  # 如果某个原子缺失，返回 None

def align_proteins(structure1, structure2):
    """
    对齐两个蛋白质结构，使用 BioPython 的 Superimposer。
    """
    atoms1 = [residue['CA'] for residue in structure1.get_residues() if is_aa(residue) and 'CA' in residue]
    atoms2 = [residue['CA'] for residue in structure2.get_residues() if is_aa(residue) and 'CA' in residue]
    
    # 使用序列比对找到最优残基对齐
    seq1 = ''.join([residue.get_resname()[:3] for residue in structure1.get_residues() if is_aa(residue)])
    seq2 = ''.join([residue.get_resname()[:3] for residue in structure2.get_residues() if is_aa(residue)])
    
    substitution_matrix = substitution_matrices.load('BLOSUM62')
    valid_residues = set([key[0] for key in substitution_matrix.keys()])
    
    # 替换无效残基为 'G'
    seq1 = ''.join([res if res in valid_residues else 'G' for res in seq1])
    seq2 = ''.join([res if res in valid_residues else 'G' for res in seq2])
    
    # 转换序列为字符串，以避免类型不匹配的问题
    alignment = pairwise2.align.globaldx(seq1, seq2, substitution_matrix)[0]
    
    optimal_pairs = []
    index1, index2 = 0, 0
    for i, j in zip(alignment.seqA, alignment.seqB):
        if i != '-' and j != '-':
            if index1 < len(atoms1) and index2 < len(atoms2):
                residue1 = atoms1[index1]
                residue2 = atoms2[index2]
                optimal_pairs.append((residue1, residue2))
        if i != '-':
            index1 += 1
        if j != '-':
            index2 += 1
    
    # 对齐两个结构，使用最优的残基配对
    if optimal_pairs:
        sup = Superimposer()
        atoms1_optimal, atoms2_optimal = [pair[0] for pair in optimal_pairs], [pair[1] for pair in optimal_pairs]
        sup.set_atoms(atoms1_optimal, atoms2_optimal)
        sup.apply(structure2.get_atoms())
        return optimal_pairs, sup.rms  # 返回对齐的残基对及 RMSD
    else:
        return [], 0.0  # 如果没有找到配对，返回空列表和 RMSD 为 0.0

def calculate_plane_angle(normal_vector1, normal_vector2):
    """
    计算两个法向量之间的夹角。
    """
    if normal_vector1 is None or normal_vector2 is None:
        return None
    dot_product = np.dot(normal_vector1, normal_vector2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def compare_residue_planes(structure1, structure2):
    """
    比较两个蛋白质中所有残基的平面法向量。
    """
    residue_pairs, rmsd = align_proteins(structure1, structure2)
    print(f"RMSD after alignment: {rmsd:.2f} Å")
    
    for atom1, atom2 in residue_pairs:
        res1, res2 = atom1.get_parent(), atom2.get_parent()
        normal_vector1, normal_vector2 = get_residue_plane_normal_vector(res1), get_residue_plane_normal_vector(res2)
        angle_degrees = calculate_plane_angle(normal_vector1, normal_vector2)
        
        if angle_degrees is not None:
            print(f"Residue {res1.id[1]} (Protein1) and Residue {res2.id[1]} (Protein2): Angle between planes = {angle_degrees:.2f} degrees")

def main():
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("protein1", "REF1.pdb")[0]['A']
    structure2 = parser.get_structure("protein2", "7xjf.pdb")[0]['A']
    
    compare_residue_planes(structure1, structure2)

if __name__ == "__main__":
    main()
