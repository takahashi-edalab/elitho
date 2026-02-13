from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
import sys
from datetime import datetime
from decimal import Decimal

try:
    from elitho import source, config, intensity
except Exception:
    _here = os.path.dirname(__file__)
    _root = os.path.abspath(os.path.join(_here, os.pardir))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from elitho import source, config, intensity


st.set_page_config(page_title="ELitho Simulation", layout="wide")
st.title("ELitho Simulation")


try:
    with open("defaults.json", "r", encoding="utf-8") as f:
        defaults = json.load(f)
except Exception:
    defaults = {
        "wavelength": 13.5,
        "NA": 0.33,
        "mask_width": 1024,
        "mask_height": 1024,
        "magnification_x": 4,
        "magnification_y": 4,
        "mesh": 0.5,
        "incidence_angle": -6.0,
        "azimuthal_angle": 0.0,
        "central_obscuration": 0.2,
        "defocus_min_nm": -100,
        "absorber_complex_refractive_index": [0.9567 + 0.0343j],
        "absorber_thickness": [60.0],
    }


def generate_mask(
    width: float, height: float, opens: list[dict[str, float]], pixels_per_um: int = 10
):
    width = max(1e-6, float(width))
    height = max(1e-6, float(height))

    width_px = max(1, int(round(width * pixels_per_um)))
    height_px = max(1, int(round(height * pixels_per_um)))
    mask = np.zeros((height_px, width_px), dtype=np.uint8)

    for op in opens:
        cx = float(op.get("center_x_um", 0.0))
        cy = float(op.get("center_y_um", 0.0))
        w = float(op.get("width_um", 0.0))
        h = float(op.get("height_um", 0.0))

        x0 = int(round((cx + width / 2.0 - w / 2.0) * pixels_per_um))
        x1 = int(round((cx + width / 2.0 + w / 2.0) * pixels_per_um))
        y0 = int(round((height / 2.0 - cy - h / 2.0) * pixels_per_um))
        y1 = int(round((height / 2.0 - cy + h / 2.0) * pixels_per_um))

        x0 = max(0, min(width_px - 1, x0))
        x1 = max(0, min(width_px, x1))
        y0 = max(0, min(height_px - 1, y0))
        y1 = max(0, min(height_px, y1))

        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] = 1

    return mask


def generate_intensities(sc: config.SimulationConfig, mask: np.ndarray) -> list:
    results = []
    for defocus in sc.defocus_list:
        result = [defocus]
        for polar in [config.PolarizationDirection.X, config.PolarizationDirection.Y]:
            intensity_result = intensity.intensity(sc, mask, polar, defocus)
            result.append(intensity_result)
        unpolar_intensity_result = (result[-1] + result[-2]) / 2.0
        result.append(unpolar_intensity_result)
        results.append(tuple(result))
    return results


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Decimal and float with precision"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

    def encode(self, obj):
        def normalize_floats(item):
            if isinstance(item, float):
                # Remove trailing zeros and unnecessary precision
                s = f"{item:.10f}".rstrip("0").rstrip(".")
                return float(s)
            elif isinstance(item, dict):
                return {k: normalize_floats(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [normalize_floats(i) for i in item]
            return item

        return super().encode(normalize_floats(obj))


def save_parameters(params: dict, filename: str = None) -> str:
    """Save all parameters to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elitho_params_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)

    return filename


def load_parameters(filename: str) -> dict:
    """Load parameters from JSON file"""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def render_inputs():
    # Save/Load controls in an expander for cleaner UI
    with st.expander("💾 Save / Load Parameters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Save Parameters**")

            # Collect all parameters
            params = {
                "wavelength": st.session_state.get(
                    "wavelength", defaults.get("wavelength", 13.5)
                ),
                "NA": st.session_state.get("NA", defaults.get("NA", 0.33)),
                "magnification_x": st.session_state.get(
                    "magnification_x", defaults.get("magnification_x", 4)
                ),
                "magnification_y": st.session_state.get(
                    "magnification_y", defaults.get("magnification_y", 4)
                ),
                "central_obscuration": st.session_state.get(
                    "central_obscuration", defaults.get("central_obscuration", 0.2)
                ),
                "incidence_angle": st.session_state.get(
                    "incidence_angle", defaults.get("incidence_angle", -6.0)
                ),
                "azimuthal_angle": st.session_state.get(
                    "azimuthal_angle", defaults.get("azimuthal_angle", 0.0)
                ),
                "defocus_min": st.session_state.get(
                    "defocus_min", defaults.get("defocus_min_um", 0.0)
                ),
                "defocus_max": st.session_state.get(
                    "defocus_max", defaults.get("defocus_max_um", 1.0)
                ),
                "defocus_step": st.session_state.get(
                    "defocus_step", defaults.get("defocus_step_um", 0.1)
                ),
                "mask_width": st.session_state.get(
                    "mask_width", defaults.get("mask_width", 1024)
                ),
                "mask_height": st.session_state.get(
                    "mask_height", defaults.get("mask_height", 1024)
                ),
                "mesh": st.session_state.get("mesh", defaults.get("mesh", 0.5)),
                "source_type": st.session_state.get("source_type", "CIRCULAR"),
                "outer_sigma": st.session_state.get("outer_sigma", 0.9),
                "inner_sigma": st.session_state.get("inner_sigma", 0.55),
                "open_angle": st.session_state.get("open_angle", 90.0),
                "num_layers": st.session_state.get("num_layers", 1),
            }

            # Add layer parameters
            num_layers = int(params["num_layers"])
            layers = []
            for li in range(num_layers):
                layers.append(
                    {
                        "n": st.session_state.get(f"layer_{li}_n", 0.9567),
                        "k": st.session_state.get(f"layer_{li}_k", 0.0343),
                        "thickness": st.session_state.get(
                            f"layer_{li}_thickness", 60.0
                        ),
                    }
                )
            params["layers"] = layers

            # Add mask opens
            num_opens = st.session_state.get("mask_num_opens", 1)
            params["num_opens"] = num_opens
            opens = []
            for oi in range(1, int(num_opens) + 1):
                opens.append(
                    {
                        "center_x_um": st.session_state.get(f"mask_open_{oi}_cx", 0.0),
                        "center_y_um": st.session_state.get(f"mask_open_{oi}_cy", 0.0),
                        "width_um": st.session_state.get(f"mask_open_{oi}_w", 50.0),
                        "height_um": st.session_state.get(f"mask_open_{oi}_h", 50.0),
                    }
                )
            params["opens"] = opens

            # Convert to JSON string for download
            json_str = json.dumps(
                params, indent=2, ensure_ascii=False, cls=DecimalEncoder
            )

            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"elitho_params_{timestamp}.json"

            st.download_button(
                label="💾 Download Parameters",
                data=json_str,
                file_name=default_filename,
                mime="application/json",
                use_container_width=True,
            )

        with col2:
            st.markdown("**Load Parameters**")

            uploaded_file = st.file_uploader(
                "Choose a JSON file", type=["json"], key="param_loader"
            )

            if uploaded_file is not None:
                # Use file name and size as unique identifier
                file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

                # Auto-load if this is a new file
                if st.session_state.get("last_loaded_file") != file_signature:
                    try:
                        # Read file content as bytes and decode
                        file_content = uploaded_file.read()
                        params = json.loads(file_content.decode("utf-8"))

                        # Load parameters into session state
                        for key, value in params.items():
                            if key == "layers":
                                st.session_state["num_layers"] = len(value)
                                for li, layer in enumerate(value):
                                    st.session_state[f"layer_{li}_n"] = layer.get(
                                        "n", 0.0
                                    )
                                    st.session_state[f"layer_{li}_k"] = layer.get(
                                        "k", 0.0
                                    )
                                    st.session_state[f"layer_{li}_thickness"] = (
                                        layer.get("thickness", 0.0)
                                    )
                            elif key == "opens":
                                st.session_state["mask_num_opens"] = len(value)
                                for oi, open_data in enumerate(value, start=1):
                                    st.session_state[f"mask_open_{oi}_cx"] = int(
                                        open_data.get("center_x_um", 0)
                                    )
                                    st.session_state[f"mask_open_{oi}_cy"] = int(
                                        open_data.get("center_y_um", 0)
                                    )
                                    st.session_state[f"mask_open_{oi}_w"] = int(
                                        open_data.get("width_um", 50)
                                    )
                                    st.session_state[f"mask_open_{oi}_h"] = int(
                                        open_data.get("height_um", 50)
                                    )
                            elif key not in ["num_opens"]:
                                st.session_state[key] = value

                        # Mark this file as loaded
                        st.session_state["last_loaded_file"] = file_signature
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                # File was cleared (X button clicked)
                if "last_loaded_file" in st.session_state:
                    st.session_state.pop("last_loaded_file", None)

    st.subheader("Optical Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wavelength = st.number_input(
            "wavelength [nm]",
            min_value=0.0,
            value=float(defaults.get("wavelength", 13.5)),
            step=0.1,
            format="%.1f",
            key="wavelength",
        )
        _na_options = [0.33, 0.55]
        _default_na = float(defaults.get("NA", 0.55))
        _default_index = (
            _na_options.index(_default_na) if _default_na in _na_options else 1
        )
        NA = st.selectbox(
            "NA",
            _na_options,
            index=_default_index,
            format_func=lambda x: f"{x:.2f}",
            key="NA",
        )
        prev_na = st.session_state.get("_prev_NA", None)
        if prev_na is None or float(prev_na) != float(NA):
            if float(NA) == 0.33:
                st.session_state["magnification_x"] = 4
                st.session_state["magnification_y"] = 4
            elif float(NA) == 0.55:
                st.session_state["magnification_x"] = 4
                st.session_state["magnification_y"] = 8
            st.session_state["_prev_NA"] = float(NA)

    with c2:
        magnification_x = st.number_input(
            "X magnification",
            min_value=0,
            value=int(defaults.get("magnification_x", 4)),
            step=1,
            format="%d",
            key="magnification_x",
        )
        magnification_y = st.number_input(
            "Y magnification",
            min_value=0,
            value=int(defaults.get("magnification_y", 4)),
            step=1,
            format="%d",
            key="magnification_y",
        )
        if float(NA) == 0.55:
            central_obscuration = st.number_input(
                "Central obscuration [fraction]",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("central_obscuration", 0.2)),
                step=0.01,
                format="%.2f",
                key="central_obscuration",
            )
        else:
            central_obscuration = 0.0

    with c3:
        incidence_angle = st.number_input(
            "incidence angle [deg]",
            value=float(defaults.get("incidence_angle", -6.0)),
            step=0.1,
            format="%.1f",
            key="incidence_angle",
        )
        azimuthal_angle = st.number_input(
            "azimuthal angle [deg]",
            value=float(defaults.get("azimuthal_angle", 0.0)),
            step=0.1,
            format="%.1f",
            key="azimuthal_angle",
        )

    with c4:
        defocus_mode = st.radio(
            "defocus mode",
            [
                "single",
                "sweep",
            ],
            horizontal=True,
        )
        if defocus_mode == "single":
            defocus_min = st.number_input(
                "defocus value [nm]", value=0, format="%d", key="defocus_single"
            )
            defocus_max = None
            defocus_step = None
        else:
            defocus_min = st.number_input(
                "defocus min [nm]",
                value=int(defaults.get("defocus_min_nm", -100)),
                step=1,
                format="%d",
                key="defocus_min",
            )
            defocus_max = st.number_input(
                "defocus max [nm]",
                value=int(defaults.get("defocus_max_nm", 100)),
                step=1,
                format="%d",
                key="defocus_max",
            )
            defocus_step = st.number_input(
                "defocus step [nm]",
                min_value=0,
                value=int(defaults.get("defocus_step_nm", 20)),
                step=1,
                format="%d",
                key="defocus_step",
            )

    st.markdown("---")

    st.subheader("Absorber Parameters")
    thickness = defaults.get("absorber_thickness", [])
    complex_refractive_index = defaults.get("absorber_complex_refractive_index", [])
    initial_n_layers = len(complex_refractive_index)
    num_layers = st.number_input(
        "Number of layers",
        min_value=1,
        max_value=20,
        value=initial_n_layers,
        step=1,
        key="num_layers",
    )
    layers = []
    for li in range(0, int(num_layers)):
        # Display the first layer as "Layer 1 (top)"; keep other layers unchanged.
        # Comments are written in English per project request.
        label = f"Layer {li + 1} (top)" if li == 0 else f"Layer {li + 1}"
        with st.expander(label, expanded=True):
            st.markdown("Complex refractive index")
            a1, a2 = st.columns(2)

            # Determine default values
            if li < len(complex_refractive_index):
                default_n = complex_refractive_index[li].real
                default_k = complex_refractive_index[li].imag
            else:
                default_n = 0.9567 if li == 0 else 0.0
                default_k = 0.0343 if li == 0 else 0.0

            if li < len(thickness):
                default_thickness = thickness[li]
            else:
                default_thickness = 60.0 if li == 0 else 0.0

            with a1:
                n_real = st.number_input(
                    "Real part (n)",
                    min_value=0.0,
                    value=default_n,
                    step=0.0001,
                    format="%.4f",
                    key=f"layer_{li}_n",
                )
            with a2:
                k_imag = st.number_input(
                    "Imag part (k)",
                    min_value=0.0,
                    value=default_k,
                    step=0.0001,
                    format="%.4f",
                    key=f"layer_{li}_k",
                )
            thickness_val = st.number_input(
                "Thickness [nm]",
                min_value=0.0,
                value=default_thickness,
                step=0.1,
                format="%.1f",
                key=f"layer_{li}_thickness",
            )
            layers.append(
                {
                    "thickness": float(thickness_val),
                    "n": float(n_real),
                    "k": float(k_imag),
                }
            )

    # Create AbsorberLayers instance from GUI parameters
    thicknesses = [layer["thickness"] for layer in layers]
    complex_refractive_indices = [complex(layer["n"], layer["k"]) for layer in layers]
    absorber_layers = config.AbsorberLayers(
        thicknesses=thicknesses, complex_refractive_indices=complex_refractive_indices
    )

    st.markdown("---")

    st.subheader("Mask Pattern Parameters")
    mp_left, mp_right = st.columns([3, 1])
    with mp_left:
        tw, th = st.columns(2)
        with tw:
            mask_width = st.number_input(
                "Mask width [nm]",
                min_value=0,
                value=int(defaults.get("mask_width", 1024)),
                step=10,
                format="%d",
                key="mask_width",
            )
        with th:
            mask_height = st.number_input(
                "Mask height [nm]",
                min_value=0,
                value=int(defaults.get("mask_height", 1024)),
                step=10,
                format="%d",
                key="mask_height",
            )
        num_opens = st.number_input(
            "Number of opens",
            min_value=0,
            max_value=100,
            value=1,
            step=1,
            key="mask_num_opens",
        )
        opens = []
        for oi in range(1, int(num_opens) + 1):
            with st.expander(f"Open {oi}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    cx = st.number_input(
                        "center X [nm]",
                        value=0,
                        step=1,
                        format="%d",
                        key=f"mask_open_{oi}_cx",
                    )
                with c2:
                    cy = st.number_input(
                        "center Y [nm]",
                        value=0,
                        step=1,
                        format="%d",
                        key=f"mask_open_{oi}_cy",
                    )
                with c3:
                    ow = st.number_input(
                        "width [nm]",
                        min_value=0,
                        value=50,
                        step=1,
                        format="%d",
                        key=f"mask_open_{oi}_w",
                    )
                with c4:
                    oh = st.number_input(
                        "height [nm]",
                        min_value=0,
                        value=50,
                        step=1,
                        format="%d",
                        key=f"mask_open_{oi}_h",
                    )
                opens.append(
                    {
                        "center_x_um": int(cx),
                        "center_y_um": int(cy),
                        "width_um": int(ow),
                        "height_um": int(oh),
                    }
                )

    with mp_right:
        try:
            mask_arr = generate_mask(mask_width, mask_height, opens, pixels_per_um=5)
            fig_mask, axm = plt.subplots(figsize=(2, 2))
            axm.imshow(mask_arr, cmap="gray", origin="lower", interpolation="nearest")
            axm.axis("off")
            st.markdown("Mask preview (white = open)")
            st.pyplot(fig_mask, width=256)
        except Exception as e:
            st.error(f"Unable to render mask preview: {e}")

    st.markdown("---")

    st.subheader("Source Parameters")
    # Mesh input: allow the user to set the sampling mesh used in calculations.
    # Default to const.mesh when available, otherwise 0.50. Display with two decimals.
    mesh = st.number_input(
        "mesh [degree]",
        min_value=0.0,
        value=float(getattr(config, "mesh", 0.5)),
        step=0.01,
        format="%.2f",
        key="mesh",
    )

    sp_left, sp_right = st.columns([2, 1])
    with sp_left:
        # Source type selection: choose illumination type for the simulation.
        # Options correspond to elitho.const.IlluminationType members.
        _illum_options = [t.name for t in config.IlluminationType]
        _default_illum = _illum_options[0]
        _default_index = (
            _illum_options.index(_default_illum)
            if _default_illum in _illum_options
            else 0
        )
        source_type_name = st.selectbox(
            "Illumination type",
            _illum_options,
            index=_default_index,
            key="source_type",
        )
        # Map selected name to the enum value for downstream code compatibility.
        try:
            source_type_enum = config.IlluminationType[source_type_name]
        except Exception:
            source_type_enum = config.IlluminationType.CIRCULAR

        # Conditional inputs depending on the chosen source type.
        # initial parameters
        outer_sigma = 0.9
        inner_sigma = 0.55
        open_angle = 90.0

        if source_type_enum == config.IlluminationType.CIRCULAR:
            outer_sigma = st.number_input(
                "Outer sigma",
                min_value=0.0,
                value=outer_sigma,
                step=0.01,
                format="%.2f",
                key="outer_sigma",
            )
            ill = config.CircularIllumination(outer_sigma=float(outer_sigma))

        elif source_type_enum == config.IlluminationType.ANNULAR:
            outer_sigma = st.number_input(
                "Outer sigma",
                min_value=0.0,
                value=outer_sigma,
                step=0.01,
                format="%.2f",
                key="outer_sigma",
            )
            inner_sigma = st.number_input(
                "Inner sigma",
                min_value=0.0,
                value=inner_sigma,
                step=0.01,
                format="%.2f",
                key="inner_sigma",
            )
            ill = config.AnnularIllumination(
                outer_sigma=float(outer_sigma),
                inner_sigma=float(inner_sigma),
            )

        elif source_type_enum in (
            config.IlluminationType.DIPOLE_X,
            config.IlluminationType.DIPOLE_Y,
        ):
            outer_sigma = st.number_input(
                "Outer sigma",
                min_value=0.0,
                value=outer_sigma,
                step=0.01,
                format="%.2f",
                key="outer_sigma",
            )
            inner_sigma = st.number_input(
                "Inner sigma",
                min_value=0.0,
                value=inner_sigma,
                step=0.01,
                format="%.2f",
                key="inner_sigma",
            )
            open_angle = st.number_input(
                "Open angle [deg]",
                min_value=0.0,
                value=open_angle,
                step=0.1,
                format="%.1f",
                key="open_angle",
            )
            ill = config.DipoleIllumination(
                type=source_type_enum,
                outer_sigma=float(outer_sigma),
                inner_sigma=float(inner_sigma),
                open_angle=float(open_angle),
            )
        else:
            raise ValueError("Unsupported illumination type")

    sc = config.SimulationConfig(
        wavelength=float(wavelength),
        NA=float(NA),
        is_high_na=NA > 0.33,
        illumination=ill,
        absorber_layers=absorber_layers,
        mask_width=int(mask_width),
        mask_height=int(mask_height),
        magnification_x=int(magnification_x),
        magnification_y=int(magnification_y),
        mesh=float(mesh),
        incidence_angle=float(incidence_angle),
        azimuthal_angle=float(azimuthal_angle),
        central_obscuration=float(central_obscuration),
        defocus_min=defocus_min,
        defocus_max=defocus_max,
        defocus_step=defocus_step,
    )

    with sp_right:
        try:
            k = 2.0 * np.pi / wavelength
            # dkx, dky, _ = source.uniform_k_source(sc, ill)
            dkx, dky, _, _, _ = source.get_valid_source_points(sc)
            sxo = dkx / k / NA
            syo = dky / k / NA
            # Visualize source directions on a small figure using subplots
            ill_fig, ill_axi = plt.subplots()
            ill_axi.set_aspect("equal", adjustable="box")
            ill_axi.plot(sxo, syo, "o")
            ill_axi.set_xlim(-1.0, 1.0)
            ill_axi.set_ylim(-1.0, 1.0)
            st.markdown("**Illumination preview**")
            st.pyplot(ill_fig, width=350)
        except Exception as e:
            st.error(f"Unable to render illumination preview: {e}")

    return sc, mask_arr


def show_intensity(title: str, img: np.ndarray) -> None:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(title)
    with col2:
        fig_mask, axm = plt.subplots()
        axm.imshow(img)
        axm.axis("off")
        st.pyplot(fig_mask, width=256)


# Layout: two columns (inputs / results)
# main_col, side_col = st.columns([6, 4], width=2048)
main_col, side_col = st.columns([6, 4], width=3072)
with main_col:
    sc, mask = render_inputs()

with side_col:
    st.subheader("Simulation Results")

    # Buttons for run and save
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run = st.button("Run simulation", use_container_width=True)
    with btn_col2:
        # Save button (only enabled if results exist)
        save_enabled = "_generated_intensity" in st.session_state
        if save_enabled:
            results = st.session_state["_generated_intensity"]

            # Prepare data for download
            defocus_values = np.array([r[0] for r in results])
            x_polarized = np.array([r[1] for r in results])
            y_polarized = np.array([r[2] for r in results])
            unpolarized = np.array([r[3] for r in results])

            # Create npz file in memory
            import io

            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                defocus=defocus_values,
                x_polarized=x_polarized,
                y_polarized=y_polarized,
                unpolarized=unpolarized,
            )
            buffer.seek(0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"elitho_results_{timestamp}.npz"

            st.download_button(
                label="💾 Save results",
                data=buffer.getvalue(),
                file_name=default_filename,
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.button("💾 Save results", disabled=True, use_container_width=True)

    result_placeholder = st.empty()

    # When Run pressed: generate intensities and persist into session_state
    if run:
        with st.spinner("Run simulation ..."):
            defocus2intensities = generate_intensities(sc, mask)
        st.session_state["_generated_intensity"] = defocus2intensities
        st.rerun()

    # If frames exist, show defocus-value slider and selected frame
    if "_generated_intensity" in st.session_state:
        results = st.session_state["_generated_intensity"]

        if len(results) == 1:
            # Single defocus case
            result = results[0]
            defocus_val, int_x_polar, int_y_polar, int_unpolar = result
            with result_placeholder.container():
                show_intensity("X Polarization", int_x_polar)
                show_intensity("Y Polarization", int_y_polar)
                show_intensity("Unpolarized", int_unpolar)
        else:
            # Multiple defocus case
            with result_placeholder.container():
                # Extract defocus values
                defocus_values = [r[0] for r in results]

                # Slider for defocus selection
                selected_defocus = st.select_slider(
                    "Defocus [nm]",
                    options=defocus_values,
                    format_func=lambda x: f"{x:.3f}",
                )

                # Find the selected result
                selected_idx = defocus_values.index(selected_defocus)
                _, int_x_polar, int_y_polar, int_unpolar = results[selected_idx]

                # Display the three polarization results
                show_intensity("X Polarization", int_x_polar)
                show_intensity("Y Polarization", int_y_polar)
                show_intensity("Unpolarized", int_unpolar)
