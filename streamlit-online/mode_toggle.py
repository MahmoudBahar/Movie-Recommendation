import streamlit as st

from data_loader import light_mode_enabled


def _get_password_secret() -> str:
    try:
        return st.secrets.get("heavy_mode_password", None)
    except Exception:
        return None


def resolve_light_mode(key_prefix: str = "") -> bool:
    """Render the light/heavy mode toggle and gate heavy mode behind a password when provided.

    key_prefix keeps Streamlit widget keys unique across pages.
    """
    light_mode_default = light_mode_enabled()
    override_key = f"{key_prefix}light_mode_override"
    unlocked_key = f"{key_prefix}heavy_mode_unlocked"
    toggle_key = f"{key_prefix}light_mode_toggle"
    pwd_input_key = f"{key_prefix}heavy_mode_pwd"
    btn_key = f"{key_prefix}heavy_mode_unlock_btn"

    if override_key not in st.session_state:
        st.session_state[override_key] = light_mode_default
    if unlocked_key not in st.session_state:
        st.session_state[unlocked_key] = False

    desired_light_mode = st.toggle(
        "Light mode (RAM saver)",
        value=st.session_state[override_key],
        help="Disable to load collaborative filtering models and ratings.",
        key=toggle_key,
    )

    password_secret = _get_password_secret()
    light_mode = desired_light_mode

    if password_secret:
        unlocked = st.session_state[unlocked_key]
        if not unlocked and not desired_light_mode:
            st.info(
                "Heavy mode is restricted here. If you need full mode without the password, "
                "clone the GitHub repo and run the app locally to disable light mode safely."
            )
            pwd = st.text_input(
                "Heavy mode password",
                type="password",
                key=pwd_input_key,
                help="Provided by the deployment secrets",
            )
            if st.button("Unlock heavy mode", key=btn_key):
                if pwd == password_secret:
                    st.session_state[unlocked_key] = True
                    light_mode = False
                    st.success("Heavy mode unlocked for this session.")
                else:
                    st.warning("Incorrect password. Staying in light mode.")
                    light_mode = True
            else:
                # Wait for user to press unlock; stay in light mode meanwhile.
                light_mode = True
        elif unlocked:
            light_mode = desired_light_mode
    else:
        light_mode = desired_light_mode

    st.session_state[override_key] = light_mode
    return light_mode
