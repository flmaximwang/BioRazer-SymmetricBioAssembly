import numpy as np
from .generator import generate_helix_ca_by_crick, generate_cc_ca_by_crick
from scipy.optimize import least_squares


def _fit_helix_by_crick_residuals(params, ca_coords_observed, N):
    centroid = params[:3]
    direction = params[3:6]
    radius = params[6]
    omega = params[7]
    pitch_angle = params[8]
    phi0 = params[9]
    pred_ca, _ = generate_helix_ca_by_crick(
        residue_num=N,
        centroid=centroid,
        direction=direction,
        radius=radius,
        omega=omega,
        pitch_angle=pitch_angle,
        phi0=phi0,
    )
    return (pred_ca - ca_coords_observed).flatten()


def fit_helix_by_crick(
    ca_coords_obs: np.ndarray,
    initial_params=dict(
        radius=2.26,
        omega=4 * np.pi / 7,
        pitch_angle=0.876,
        phi0=0.0,
    ),
):
    """
    Fit a Crick helix to the observed CA coordinates.

    Parameters:
    -----------------
    - observed_ca: np.ndarray - The coordinates of the CA atoms to fit.
    - initial_params: dict - Initial parameters for the helix fitting.


    Returns:
    -----------------
    - params: dict - Fitted parameters of the helix.
    - rmsd
    - xyz: np.ndarray - The coordinates of the fitted helix CA atoms.
    """

    initial_params["centroid"] = np.mean(ca_coords_obs, axis=0)
    initial_params["direction"] = ca_coords_obs[-1] - ca_coords_obs[0]
    p0 = np.array(
        [
            *initial_params["centroid"],
            *initial_params["direction"] / np.linalg.norm(initial_params["direction"]),
            initial_params["radius"],
            initial_params["omega"],
            initial_params["pitch_angle"],
            initial_params["phi0"],
        ]
    )
    lb_params = dict(
        centroid=(-np.inf, -np.inf, -np.inf),
        direction=(-1.0, -1.0, -1.0),
        radius=0.0,
        omega=0.0,
        pitch_angle=-np.inf,
        phi0=-np.pi,
    )
    lb = np.array(
        [
            *lb_params["centroid"],
            *lb_params["direction"],
            lb_params["radius"],
            lb_params["omega"],
            lb_params["pitch_angle"],
            lb_params["phi0"],
        ]
    )
    ub_params = dict(
        centroid=(np.inf, np.inf, np.inf),
        direction=(1.0, 1.0, 1.0),
        radius=np.inf,
        omega=np.inf,
        pitch_angle=np.inf,
        phi0=np.pi,
    )
    ub = np.array(
        [
            *ub_params["centroid"],
            *ub_params["direction"],
            ub_params["radius"],
            ub_params["omega"],
            ub_params["pitch_angle"],
            ub_params["phi0"],
        ]
    )

    N = ca_coords_obs.shape[0]
    result = least_squares(
        _fit_helix_by_crick_residuals,
        p0,
        bounds=(lb, ub),
        args=(ca_coords_obs, N),
        method="trf",
    )

    if not result.success:
        raise ValueError("Crick helix fitting failed: " + result.message)
    params = result.x
    centroid = params[:3]
    direction = params[3:6]
    direction /= np.linalg.norm(direction)
    if direction @ initial_params["direction"] < 0:
        direction = -direction
    radius = params[6]
    omega = params[7]
    pitch_angle = params[8]
    phi0 = params[9]
    rmsd = np.sqrt(np.sum(result.fun**2) / N)
    xyz, params = generate_helix_ca_by_crick(
        N, centroid, direction, radius, omega, pitch_angle, phi0
    )

    return params, rmsd, xyz


def _fit_sym_cc_by_crick_residual(params, ca_coords_obs, parse_dict, **kwargs):
    params_kwargs = {}
    for key, value in parse_dict.items():
        params_kwargs[key] = params[value]
    xyz_pred, _ = generate_cc_ca_by_crick(
        **params_kwargs,
        **kwargs,
    )
    return (xyz_pred - ca_coords_obs).flatten()


def _construct_param_vector(param_dict, param_names):
    """
    Construct a parameter vector from the parameter dictionary based on the specified names.
    """
    param_vector = []
    for name in param_names:
        if isinstance(param_dict[name], (int, float)):
            param_vector.append(param_dict[name])
        else:
            param_vector.extend(param_dict[name])
    return np.array(param_vector)


def _optimize_sym_cc_by_crick(ca_coords_obs, initial_params, param_names_to_optimize):

    lb_params = dict(
        centroid=(-np.inf, -np.inf, -np.inf),
        direction=(-1.0, -1.0, -1.0),
        r0=2.5,
        w0=-np.pi,
        phi0=-np.pi,
        r1s=1.2,
        w1s=2 * np.pi / 7,
        phi1s=-np.pi,
        pitch_angles=-np.pi,
        z_offsets=-np.inf,
    )
    ub_params = dict(
        centroid=(np.inf, np.inf, np.inf),
        direction=(1.0, 1.0, 1.0),
        r0=10,
        w0=np.inf,
        phi0=np.pi,
        r1s=4.6,
        w1s=np.pi,
        phi1s=np.pi,
        pitch_angles=np.pi,
        z_offsets=np.inf,
    )

    p0 = _construct_param_vector(initial_params, param_names_to_optimize)
    lb = _construct_param_vector(lb_params, param_names_to_optimize)
    ub = _construct_param_vector(ub_params, param_names_to_optimize)

    # print("============== Initial parameters: ================")
    # for key, value in initial_params.items():
    #     print(f"{key}: {value}")
    # print("============== Optimizing parameters below: =======")
    # print(param_names_to_optimize)
    # print("================== Checking bounds: ===============")
    # print((p0 - lb) > 0)
    # print((ub - p0) > 0)

    fixed_params = initial_params.copy()
    for name in param_names_to_optimize:
        fixed_params.pop(name, None)
    parse_dict = {}
    param_index = 0
    for name in param_names_to_optimize:
        if isinstance(initial_params[name], (int, float)):
            parse_dict[name] = param_index
            param_index += 1
        else:
            parse_dict[name] = slice(
                param_index, param_index + len(initial_params[name])
            )
            param_index += len(initial_params[name])
    kwargs = dict(parse_dict=parse_dict)
    kwargs.update(fixed_params)
    result = least_squares(
        _fit_sym_cc_by_crick_residual,
        p0,
        bounds=(lb, ub),
        args=[ca_coords_obs],
        kwargs=kwargs,
        method="trf",
    )
    if not result.success:
        raise ValueError("Coiled-coil helix fitting failed: " + result.message)
    params = result.x
    optimized_params = {}
    for name, value in parse_dict.items():
        optimized_params[name] = params[value]
    final_params = optimized_params.copy()
    final_params.update(fixed_params)
    final_params["direction"] /= np.linalg.norm(final_params["direction"])
    return final_params, result


def fit_sym_cc_by_crick(
    ca_coords_obs: np.ndarray,
    initial_params=dict(
        r0=5.0,
        w0=-2 * np.pi / 100,
        phi0=0.0,
        r1s=2.26,
        w1s=4 * np.pi / 7,
        phi1s=0.0,
        pitch_angles=-0.2096,
        z_offsets=0,
    ),
):
    """
    Fit a coiled-coil helix to the observed CA coordinates.

    Parameters:
    -----------------
    - ca_coords_obs: np.ndarray - The coordinates of the CA atoms to fit.
        - shape: (helix_num, residue_num, 3)
    - helix_num: int - The number of helices in the coiled-coil.
    - initial_params: dict - Initial parameters for the coiled-coil fitting.

    Returns:
    -----------------
    - params: dict - Fitted parameters of the coiled-coil helix.
    - rmsd: float - Root Mean Square Deviation of the fit.
    - xyz: np.ndarray - The coordinates of the fitted coiled-coil CA atoms.
    """

    helix_num, residue_num, _ = ca_coords_obs.shape
    initial_params["helix_num"] = helix_num
    initial_params["dphi0s"] = np.linspace(0, 2 * np.pi, helix_num, endpoint=False)
    initial_params["residue_num"] = residue_num
    initial_params["centroid"] = np.mean(ca_coords_obs, axis=(0, 1))
    directions_approx = []
    for i in range(0, helix_num):
        directions_approx.append(ca_coords_obs[i, -1] - ca_coords_obs[i, 0])
    senses = [1]
    for i in range(1, helix_num):
        if np.dot(directions_approx[i], directions_approx[0]) < 0:
            senses.append(-1)
        else:
            senses.append(1)
    initial_params["senses"] = senses
    directions_with_senses = np.vstack(
        [directions_approx[i] * senses[i] for i in range(helix_num)]
    )
    initial_params["direction"] = np.mean(directions_with_senses, axis=0)
    initial_params["direction"] /= np.linalg.norm(initial_params["direction"])

    # Stage 1: Fit the centroid and direction of the coiled-coil
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=initial_params,
        param_names_to_optimize=["centroid", "direction"],
    )
    if stage_params["direction"] @ initial_params["direction"] < 0:
        stage_params["direction"] = -stage_params["direction"]

    # Stage 2: Fit r0, w0, phi0 at the same time
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=stage_params,
        param_names_to_optimize=["centroid", "direction", "r0"],
    )
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=stage_params,
        param_names_to_optimize=["centroid", "direction", "r0", "w0", "phi0"],
    )

    # Stage 3: Fit phi1s, pitch_angles, z_offsets at the same time
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=stage_params,
        param_names_to_optimize=[
            "centroid",
            "direction",
            "r0",
            "w0",
            "phi0",
            "phi1s",
            "z_offsets",
        ],
    )

    # Stage 4: Fit r1s and w1s at the same time
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=stage_params,
        param_names_to_optimize=[
            "centroid",
            "direction",
            "r0",
            "w0",
            "phi0",
            "phi1s",
            "z_offsets",
            "pitch_angles",
            "r1s",
            "w1s",
        ],
    )

    # Stage 5: 重新拟合 centroid, phi0 和 z_offsets
    stage_params["centroid"] = initial_params["centroid"]
    stage_params["phi0"] = 0
    stage_params["z_offsets"] = 0
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=stage_params,
        param_names_to_optimize=["centroid", "phi0"],
    )
    stage_params, result = _optimize_sym_cc_by_crick(
        ca_coords_obs=ca_coords_obs,
        initial_params=stage_params,
        param_names_to_optimize=[
            "centroid",
            "phi0",
            "z_offsets",
        ],
    )

    rmsd = np.sqrt(np.sum(result.fun**2) / (helix_num * residue_num))
    xyz, params = generate_cc_ca_by_crick(**stage_params)
    return params, rmsd, xyz
