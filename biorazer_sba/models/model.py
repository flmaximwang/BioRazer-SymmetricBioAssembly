import biotite.structure as bio_struct
from biotite.structure.io import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..utils.alignment import calculate_rotation, calculate_euler_ZXZ


class AssemblyPart:

    def __init__(self, structure: bio_struct.AtomArray):
        self.structure = structure

    @staticmethod
    def from_pdb(pdb_filename):
        """Load the structure from a PDB file."""
        pdb_file = pdb.PDBFile.read(pdb_filename)
        return AssemblyPart(pdb_file.get_structure()[0])

    def to_pdb(self, output_filename):
        """Export the structure to a PDB file."""
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, self.structure)
        pdb_file.write(output_filename)

    def copy(self):
        """Create a copy of the AssemblyPart."""
        return type(self)(self.structure.copy())

    def calculate_centroid(self):
        """Calculate the centroid of the structure."""
        ca_atoms = self.structure[self.structure.get_annotation("atom_name") == "CA"]
        centroid = bio_struct.centroid(ca_atoms)
        return centroid

    def calculate_x(self):
        x, _, _ = self.calculate_xyz()
        return x

    def calculate_y(self):
        _, y, _ = self.calculate_xyz()
        return y

    def calculate_z(self):
        _, _, z = self.calculate_xyz()
        return z

    def calculate_xy(self):
        x, y, _ = self.calculate_xyz()
        return x, y

    def calculate_yz(self):
        _, y, z = self.calculate_xyz()
        return y, z

    def calculate_xz(self):
        x, _, z = self.calculate_xyz()
        return x, z

    def calculate_xyz(self):
        """
        Implementation of this method should return the x, y, z directions of the structure.
        This is a placeholder method and should be implemented in subclasses.
        :return: x, y, z directions as numpy arrays.
        """
        pass

    def translate(self, x, y, z):
        self.structure = bio_struct.translate(self.structure, [x, y, z])

    def check_xz_aligned(self, atol=1e-3):
        """
        Check if the structure is aligned with the X and Z axes.
        Returns True if aligned, False otherwise.
        """
        x, z = self.calculate_xz()
        flags = [
            np.allclose(x, [1, 0, 0], atol=atol),
            np.allclose(z, [0, 0, 1], atol=atol),
        ]
        if not np.all(flags):
            raise ValueError(
                "Structure must be aligned with X and Z axes before ZXZ rotation\n"
                f"Current x: {x}\n"
                f"Current z: {z}"
            )

    def rotate_euler_ZXZ(self, a, b, c, degrees=False):
        """
        Rotate the structure with euler angles (a, b, c) in ZXZ convection. a, b, c are provided in degrees.
        """
        center_rotation = self.calculate_center_rotation()
        center_euler_xyz = center_rotation.as_euler("xyz", degrees=False)
        center_translation = self.calculate_center_translation()
        self.structure = bio_struct.translate(self.structure, center_translation)
        self.structure = bio_struct.rotate(self.structure, center_euler_xyz)

        # Now we can safely rotate the structure with ZXZ angles
        rotation = R.from_euler("ZXZ", [a, b, c], degrees=degrees)
        angles_xyz = rotation.as_euler("xyz", degrees=False)
        self.structure = bio_struct.rotate(self.structure, angles_xyz)

        # Now we need to transform the structure back to the original position
        inv_center_rotation = center_rotation.inv()
        inv_center_euler_xyz = inv_center_rotation.as_euler("xyz", degrees=False)
        inv_center_translation = -center_translation
        self.structure = bio_struct.rotate(self.structure, inv_center_euler_xyz)
        self.structure = bio_struct.translate(self.structure, inv_center_translation)

    def rotate_euler_xyz(self, a, b, c, degrees=False):
        """
        Rotate the structure with euler angles (a, b, c) in xyz convention. a, b, c are provided in degrees.
        """
        rotation = R.from_euler("xyz", [a, b, c], degrees=degrees)
        angles_xyz = rotation.as_euler("xyz", degrees=False)
        self.structure = bio_struct.rotate(self.structure, angles_xyz)

    def calculate_center_rotation(self):
        """
        Calculate the rotation that aligns the structure with the X and Z axes.
        Returns the angles (a, b, c) in degrees.
        """
        x, y, z = self.calculate_xyz()
        return calculate_rotation(x, y, z).inv()

    def calculate_center_translation(self):
        """
        Calculate the translation that moves the structure to the origin.
        Returns the translation vector.
        """
        centroid = self.calculate_centroid()
        return -centroid

    def center(self):
        counter = 0
        while not np.allclose(
            self.calculate_xyz(),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            atol=1e-5,
        ):
            counter += 1
            if counter > 10:
                raise TimeoutError(
                    "Failed to center the part after 10 attempts. "
                    "Please check the structure and alignment."
                )
            rotation = self.calculate_center_rotation()
            xyz_rotation = rotation.as_euler("xyz", degrees=False)
            translation = self.calculate_center_translation()
            self.translate(*translation)
            self.rotate_euler_xyz(*xyz_rotation)


class Assembly:

    def __init__(self, parts: list[AssemblyPart]):
        self.parts = parts

    def copy(self):
        """Create a copy of the Assembly."""
        return type(self)([part.copy() for part in self.parts])

    def append(self, new_part):
        """Append a new part to the assembly."""
        self.parts.append(new_part)

    @staticmethod
    def from_pdbs(part_type, pdb_file_paths, **kwargs):
        """Load the structure from multiple PDB files."""
        parts = []
        for filename in pdb_file_paths:
            part = part_type.from_pdb(filename, **kwargs)
            parts.append(part)
        return Assembly(parts)

    def to_pdbs(self, output_file_stem):
        """Export the structure to multiple PDB files"""
        for i, part in enumerate(self.parts):
            part.to_pdb(f"{output_file_stem}_{i}.pdb")

    def merge_to_pdb(self, output_filename):
        """Merge all parts into a single PDB file."""
        merged_structure = bio_struct.concatenate(
            [part.structure for part in self.parts]
        )
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, merged_structure)
        pdb_file.write(output_filename)

    def check_part_index(self, part_index):
        """Check if the part index is valid."""
        if part_index < 0 or part_index >= len(self.parts):
            raise IndexError("Part index out of range")

    def calculate_centroid(self, part_index):
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        return part.calculate_centroid()

    def calculate_z_direction(self, part_index):
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        return part.calculate_z()

    def calculate_x_direction(self, part_index):
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        return part.calculate_x()

    def translate(self, part_index, x, y, z):
        """Translate a specific part by (x, y, z)."""
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        part.translate(x, y, z)

    def rotate_euler(self, part_index, axis_spec, euler_angles, degrees=False):
        """
        Rotate all parts with euler angles (a, b, c) in the specified axis convention.
        axis_spec can be 'ZXZ' or 'xyz'.
        """
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        part.rotate_euler(axis_spec, *euler_angles, degrees=degrees)

    def rotate_euler_ZXZ(self, part_index, a, b, c, degrees=False):
        """Rotate a specific part with euler angles (a, b, c)."""
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        part.rotate_euler_ZXZ(a, b, c, degrees=degrees)

    def rotate_euler_xyz(self, part_index, a, b, c, degrees=False):
        """Rotate a specific part with euler angles (a, b, c) in xyz convention."""
        self.check_part_index(part_index)
        part: AssemblyPart = self.parts[part_index]
        rotation = R.from_euler("xyz", [a, b, c], degrees=degrees)
        angles_xyz = rotation.as_euler("xyz", degrees=False)
        part.structure = bio_struct.rotate(part.structure, angles_xyz)

    def center(self, part_index):

        counter = 0
        while not np.allclose(
            self.parts[part_index].calculate_xyz(),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            atol=1e-4,
        ):
            counter += 1
            if counter > 10:
                raise TimeoutError(
                    "Failed to center the part after 10 attempts. "
                    "Please check the structure and alignment."
                )
            center_part: AssemblyPart = self.parts[part_index]
            center_translation = center_part.calculate_center_translation()
            center_rotation = center_part.calculate_center_rotation()
            center_euler_xyz = center_rotation.as_euler("xyz", degrees=False)
            for part in self.parts:
                part.translate(*center_translation)
                part.rotate_euler_xyz(*center_euler_xyz)

    def calculate_translation_between(self, part_index_1, part_index_2):
        """Calculate the translation that moves part_index_1 to part_index_2."""
        self.check_part_index(part_index_1)
        self.check_part_index(part_index_2)
        part_1: AssemblyPart = self.parts[part_index_1]
        part_2: AssemblyPart = self.parts[part_index_2]
        centroid_1 = part_1.calculate_centroid()
        centroid_2 = part_2.calculate_centroid()
        return centroid_2 - centroid_1

    def calculate_ZXZ_euler_between(self, part_index_1, part_index_2, degrees=False):
        """
        Calculate the ZXZ rotation that aligns part_index_1 with part_index_2.
        Returns the angles (a, b, c) in degrees.
        """
        self.check_part_index(part_index_1)
        self.check_part_index(part_index_2)
        part_1: AssemblyPart = self.parts[part_index_1]
        part_1.check_xz_aligned()
        part_2: AssemblyPart = self.parts[part_index_2]
        x, y, z = part_2.calculate_xyz()
        rotation = R.from_matrix(np.column_stack((x, y, z)))
        angles = rotation.as_euler("ZXZ", degrees=degrees)
        return angles
