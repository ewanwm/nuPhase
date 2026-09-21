N_AVOGADRO = 6.02214076e23


class Molecule:

    def __init__(self, n_nucleons: int, molar_mass: float, name: str):

        self.n_nucleons = n_nucleons
        self.molar_mass = molar_mass
        self.name       = name

    def get_n_nucleons(self, mass: float = None):

        if mass is None:
            return self.n_nucleons

        return self.n_nucleons * mass * N_AVOGADRO / self.molar_mass

    def __str__(self):

        ret_str = f"Molecule: {self.name} with N nucleons = {self.n_nucleons} :: molar mass = {self.molar_mass}"

        return ret_str


carbon = Molecule(12, 12e-3, "Carbon")

oxygen = Molecule(16, 16e-3, "Oxygen")

water = Molecule(18, 18e-3, "Water")

def material_from_name(name: str) -> Molecule:

    if name == "carbon":
        return carbon
    elif name == "oxygen":
        return oxygen
    elif name == "water":
        return water
    else:
        raise ValueError(f"Unknown material name: {name}")

