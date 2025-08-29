from ..util.helix import direction
from .basic import *
from ..util import alignment
from biorazer.structure.io import PDB2STRUCT
from biorazer_sba.util.helix.cccp.fitter import fit_helix_by_crick, fit_sym_cc_by_crick


class AssemblyPartSingleHelix(AssemblyPart):

    def __init__(
        self,
        structure: bio_struct.AtomArray,
        helix: tuple[int, int],
    ):

        super().__init__(structure)
        assert (
            len(helix) == 2
        ), "AssemblyPartSingleHelix must specify the start and end of the helix."
        self.helix = helix

    @staticmethod
    def from_pdb(pdb_filepath, helix):
        """
        Load the structure from a PDB file and create an AssemblyPartSingleHelix instance.

        :param pdb_filepath: The path to the PDB file.
        :param helix: A tuple containing the start and end indices of the helix.
        :return: An instance of AssemblyPartSingleHelix.
        """
        structure = pdb.get_structure(pdb.PDBFile.read(pdb_filepath))[0]
        return AssemblyPartSingleHelix(structure, helix)

    def copy(self):
        return type(self)(self.structure.copy(), self.helix)

    def get_helix(self):
        """
        Get the helix structure based on the defined helix range.

        :return: An AtomArray representing the helix structure.
        """
        res_mask = np.isin(
            self.structure.get_annotation("res_id"),
            list(range(self.helix[0], self.helix[1] + 1)),
        )
        return self.structure[res_mask]

    def fit_helix_by_crick(self):
        """
        Fit a Crick helix to the CA coordinates of the helix.

        :return: A tuple containing the fitted parameters, RMSD, and the coordinates of the fitted helix CA atoms.
        """
        helix = self.get_helix()
        ca_mask = helix.get_annotation("atom_name") == "CA"
        ca_atoms = helix[ca_mask]
        ca_coords = ca_atoms.coord
        params, rmsd, xyz = fit_helix_by_crick(ca_coords)
        self.params = params
        self.rmsd = rmsd
        self.ca_coords = xyz
        z = params["direction"]
        x_prototype = ca_coords[0] - params["centroid"]
        y = np.cross(z, x_prototype)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        self.xyz = np.array([x, y, z])

    def clean_crick_params(self):
        """
        Clean up the Crick fitting parameters to free memory.
        """
        for attr_name in ["xyz", "centroid", "params", "ca_coords", "centroid"]:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def calculate_centroid(self):
        helix = self.get_helix()
        ca_mask = helix.get_annotation("atom_name") == "CA"
        ca_atoms = helix[ca_mask]
        return bio_struct.centroid(ca_atoms)

    def calculate_xyz(self):
        """
        Calculate the x, y, z axes based on the helix direction and the plane defined by the helix.

        :return: A tuple of three numpy arrays representing the x, y, and z axes.
        """
        if not hasattr(self, "xyz"):
            self.fit_helix_by_crick()
        return self.xyz

    def translate(self, x, y, z):
        super().translate(x, y, z)
        self.clean_crick_params()

    def rotate_euler_ZXZ(self, a, b, c, degrees=False):
        super().rotate_euler_ZXZ(a, b, c, degrees)
        self.clean_crick_params()

    def rotate_euler_xyz(self, a, b, c, degrees=False):
        super().rotate_euler_xyz(a, b, c, degrees)
        self.clean_crick_params()


class AssemblyPartHelixBundle(AssemblyPart):
    """
    A class representing a part of an assembly that is a pure helix.
    It inherits from AssemblyPart and provides specific functionality for helix parts.

    Parameters
    ----------
    structure: AtomArray
        The AtomArray representing the structure of the helix.
    helix_starts_ends: list[tuple[int, int]]
        A list of tuples, each containing the start and end indices of the helices in the structure.
    cc_helix_indices: list[int]
        A list of indices indicating which helices are part of the coiled coil structure.
        This is used for fitting the coiled coil structure using the CCCP model.

    """

    def __init__(
        self,
        structure: bio_struct.AtomArray,
        helix_starts_ends: list[tuple[int, int]],
        cc_helix_indices=list[int],
    ):
        super().__init__(structure)
        assert (
            len(helix_starts_ends) > 1
        ), "AssemblyPartHelixBundle must have at least two helices."

        self.helix_starts_ends = helix_starts_ends
        self.cc_helix_indices = cc_helix_indices

    @staticmethod
    def from_pdb(pdb, **kwargs):
        """
        Load the structure from a PDB file and create an AssemblyPartHelixBundle instance.

        :param pdb_filepath: The path to the PDB file.
        :param helices: A list of tuples, each containing the start and end indices of the helices.
        :param senses: A list of 1 and -1 indicating the sense of each helix.
        :param plane_helix_indices: A list of two indices indicating which helices define the plane.
        :return: An instance of AssemblyPartHelixBundle.
        """
        structure = PDB2STRUCT(pdb, "").read()
        return AssemblyPartHelixBundle(structure, **kwargs)

    def copy(self):
        return type(self)(
            self.structure.copy(),
            self.helix_starts_ends.copy(),
            self.cc_helix_indices,
        )

    def get_helix(self, helix_index: int):
        """
        Get the helix structure based on the defined helices.

        :return: An AtomArray representing the helix structure.
        """
        res_mask = np.isin(
            self.structure.get_annotation("res_id"),
            list(
                range(
                    self.helix_starts_ends[helix_index][0],
                    self.helix_starts_ends[helix_index][1] + 1,
                )
            ),
        )
        return self.structure[res_mask]

    def fit_sym_cc_by_crick(self):
        start, end = self.helix_starts_ends[self.cc_helix_indices[0]]
        helix_len = end - start
        for i in range(1, len(self.cc_helix_indices)):
            start, end = self.helix_starts_ends[self.cc_helix_indices[i]]
            helix_len_temp = end - start
            if helix_len_temp != helix_len:
                raise ValueError(
                    "Coiled coil helices must have the same length in order to be fitted by the CCCP model\n"
                    f"Current helices: {self.helix_starts_ends}"
                )
        ca_coords_cc = []
        for helix_i in self.cc_helix_indices:
            helix = self.get_helix(helix_i)
            ca_mask = helix.get_annotation("atom_name") == "CA"
            ca_atoms = helix[ca_mask]
            ca_coords_cc.append(ca_atoms.coord)
        ca_coords_cc = np.array(ca_coords_cc)
        params, rmsd, xyz = fit_sym_cc_by_crick(ca_coords_obs=ca_coords_cc)
        self.rmsd = rmsd
        self.params = params

        helix_1 = self.get_helix(self.cc_helix_indices[0])
        helix_1_centroid = bio_struct.centroid(helix_1)
        y_prototype = helix_1_centroid - self.calculate_centroid()

        z = params["direction"]
        x = np.cross(y_prototype, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.xyz = np.array([x, y, z])
        self.ca_coords_cc = xyz

    def clean_crick_params(self):
        for attr_name in ["xyz", "centroid", "parmas", "ca_coords_cc"]:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def calculate_centroid(self):
        for i in self.cc_helix_indices:
            helix = self.get_helix(i)
            if i == self.cc_helix_indices[0]:
                centroid = bio_struct.centroid(helix)
            else:
                centroid += bio_struct.centroid(helix)
        centroid /= len(self.cc_helix_indices)
        return centroid

    def calculate_helix_direction(self, helix_index: int) -> np.ndarray:
        """
        Get the direction of a specific helix based on its index.

        :param helix_index: The index of the helix for which to get the direction.
        :return: A normalized vector representing the direction of the helix.
        """
        helix = self.get_helix(helix_index)
        params, rmsd, xyz = fit_helix_by_crick(helix.coord)
        direction = params["direction"]
        direction /= np.linalg.norm(direction)
        return direction

    def calculate_xyz(self):

        if not hasattr(self, "xyz"):
            self.fit_sym_cc_by_crick()
        x, y, z = self.xyz
        return x, y, z

    def translate(self, x, y, z):
        super().translate(x, y, z)
        self.clean_crick_params()

    def rotate_euler_xyz(self, a, b, c, degrees=False):
        super().rotate_euler_xyz(a, b, c, degrees)
        self.clean_crick_params()

    def rotate_euler_ZXZ(self, a, b, c, degrees=False):
        super().rotate_euler_ZXZ(a, b, c, degrees)
        self.clean_crick_params()


class AssemblyPartHB2(AssemblyPartHelixBundle):
    """
    A class representing a part of an assembly that contains two helices.
    It inherits from AssemblyPartPureHelix and provides specific functionality for parts with two helices.
    """

    def __init__(self, structure: bio_struct.AtomArray, helices: list[tuple[int, int]]):
        """
        Initialize the AssemblyPart2Helix with a structure and a list of helices.

        :param structure: The AtomArray representing the structure of the helices.
        :param helices: A list of tuples, each containing the start and end indices of the two helices in the structure.
        """
        super().__init__(structure, helices)
        assert len(helices) == 2, "AssemblyPart2Helix must have exactly two helices."

    @staticmethod
    def from_pdb(pdb_filepath, helices):
        """
        Load the structure from a PDB file and create an AssemblyPart2Helix instance.

        :param pdb_file: The path to the PDB file.
        :param helices: A list of tuples, each containing the start and end indices of the two helices.
        :return: An instance of AssemblyPart2Helix.
        """
        structure = pdb.get_structure(pdb.PDBFile.read(pdb_filepath))[0]
        return AssemblyPartHB2(structure, helices)
