# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Streamlit GUI for Symbolic Regression

Run with: streamlit run symbolic_regression/gui_sr.py

Provides real-time visualization of the symbolic regression process including:
- Stage A separability search progression
- Stage B analytic refinement
- AST structure visualization
- Loss curves and metrics
- Feature discovery results
- Candidate acceptance/rejection tracking
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# Import SR components
from symbolic_regression.sr_runner import SRProgress, SRRunner
from symbolic_regression.sr_search.ast_viz import (
    format_loss_value,
    get_stage_emoji,
    get_ytransform_emoji,
)

# Page config
st.set_page_config(
    page_title="Symbolic Regression GUI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .reject-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and status indicator
if "sr_running" in st.session_state and st.session_state.sr_running:
    # Show line count in title for debugging
    line_count = st.session_state.get("debug_line_count", 0)
    st.title(f"🔬 Symbolic Regression GUI 🔴 LIVE ({line_count} lines)")
else:
    st.title("🔬 Symbolic Regression GUI")
st.markdown("---")

# Initialize session state
if "sr_runner" not in st.session_state:
    st.session_state.sr_runner = None
if "sr_progress" not in st.session_state:
    st.session_state.sr_progress = SRProgress()
if "sr_running" not in st.session_state:
    st.session_state.sr_running = False
if "data_filepath" not in st.session_state:
    st.session_state.data_filepath = "data/pb010_I_12_5_data.csv"  # Default to pb010
if "loss_history_data" not in st.session_state:
    st.session_state.loss_history_data = []

# Sidebar: Configuration
st.sidebar.header("⚙️ Configuration")

# Data file selection
st.sidebar.subheader("📁 Data File")
data_file = st.sidebar.text_input(
    "Data filepath:",
    value=st.session_state.data_filepath,
    help="Path to CSV file with training data",
)
if data_file != st.session_state.data_filepath:
    st.session_state.data_filepath = data_file

# Hyperparameter sections
with st.sidebar.expander("🧠 Model Hyperparameters", expanded=False):
    model_base_name = st.selectbox("Model type", ["G_Model"], index=0)
    num_segments_max = st.number_input("Max segments", value=48, min_value=1)
    num_segments_min = st.number_input("Min segments", value=16, min_value=1)
    model_size_target = st.number_input("Target params", value=1000, min_value=10)
    gmodel_scale = st.number_input("Model scale", value=0.1, format="%.3f")

with st.sidebar.expander("📊 Data Hyperparameters", expanded=False):
    batch_size = st.number_input("Batch size", value=2000, min_value=1)
    ndata_select = st.number_input("Train samples", value=2000, min_value=1)
    ndata_select_val = st.number_input("Val samples", value=2000, min_value=1)

with st.sidebar.expander("🎯 LM Hyperparameters", expanded=False):
    epochs = st.number_input("Max epochs", value=20000, min_value=1)
    epochs_min = st.number_input("Min epochs", value=1000, min_value=1)
    strategy = st.selectbox("Strategy", ["direct_solve", "explicit", "matfree"], index=0)
    loss_target = st.number_input("Loss target", value=1e-7, format="%.2e")
    loss_acceptable = st.number_input("Loss acceptable", value=1e-3, format="%.2e")
    chisq_tol = st.number_input("ChiSq tolerance", value=1e-10, format="%.2e")

with st.sidebar.expander("🔍 Search Hyperparameters", expanded=False):
    ntrial = st.number_input("Num trials", value=1, min_value=1)
    acceptance_criterion = st.number_input("Accept criterion", value=10.0, min_value=0.1)
    precision_derivs = st.number_input("Deriv precision", value=0.001, format="%.4f")

st.sidebar.markdown("---")

# Control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    start_button = st.button(
        "▶️ Start", disabled=st.session_state.sr_running, use_container_width=True
    )
with col2:
    stop_button = st.button(
        "⏹️ Stop", disabled=not st.session_state.sr_running, use_container_width=True
    )

# Handle button clicks
if start_button and not st.session_state.sr_running:
    # Create and start runner
    st.session_state.sr_runner = SRRunner(st.session_state.data_filepath)
    st.session_state.sr_runner.start()
    st.session_state.sr_running = True
    st.session_state.loss_history_data = []
    st.success("✅ SR started! Switch to **📜 Output Log** tab to watch live output!")
    time.sleep(1)  # Give user time to see the message
    st.rerun()

if stop_button and st.session_state.sr_running:
    # Stop runner
    if st.session_state.sr_runner:
        st.session_state.sr_runner.stop()
    st.session_state.sr_running = False
    st.rerun()

# Update progress if running
if st.session_state.sr_running and st.session_state.sr_runner:
    # Check if process is still running
    if not st.session_state.sr_runner.is_running():
        st.session_state.sr_running = False
    # Always update progress from runner
    st.session_state.sr_progress = st.session_state.sr_runner.get_progress()

    # Debug: show line count in title during development
    output_lines = st.session_state.sr_runner.get_output_lines()
    if output_lines:
        st.session_state.debug_line_count = len(output_lines)
    else:
        st.session_state.debug_line_count = 0

# Main content area
progress = st.session_state.sr_progress

# Top-level metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    stage_emoji = get_stage_emoji(progress.stage)
    st.metric("Stage", f"{stage_emoji} {progress.stage}")
with col2:
    elapsed_str = f"{progress.elapsed_hours * 60:.1f}m" if progress.elapsed_hours > 0 else "0m"
    st.metric("Elapsed", elapsed_str)
with col3:
    st.metric("Best Loss", format_loss_value(progress.best_val_loss))
with col4:
    st.metric("Leaves", progress.num_leaves)
with col5:
    st.metric("Parameters", f"{progress.params:,}")

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📊 Progress Details", "📜 Output Log"])

with tab1:
    st.subheader("Overview")

    if progress.stage == "init":
        st.info(
            "👋 Welcome! Configure settings in the sidebar and click Start to begin symbolic regression."
        )
        st.markdown("### Quick Start")
        st.markdown("""
        **Default Example: pb010 (x0 * x1)**
        - Simple multiplicative separability
        - Converges in ~2 minutes
        - Perfect for testing the GUI!

        **Steps:**
        1. Click **▶️ Start** in the sidebar
        2. Watch real-time progress in the tabs
        3. See the final expression when complete
        """)

    elif progress.stage == "stageA":
        # Live indicator with trial info
        trial_info = ""
        if progress.max_trial > 0:
            trial_info = f" - Trial {progress.trial}/{progress.max_trial}"
        dual_info = " (Dual Layer)" if progress.dual_layer else ""
        st.markdown(f"### 🔴 LIVE: Stage A - Separability Search{trial_info}{dual_info}")

        # Show current model prominently
        if progress.model_expression:
            st.markdown(f"### Current Model: `{progress.model_expression}`")

        col1, col2 = st.columns(2)
        with col1:
            ytrans_emoji = get_ytransform_emoji(progress.ytransform or "identity")
            st.markdown(f"**Y-Transform:** {ytrans_emoji} {progress.ytransform or 'identity'}")
            st.markdown(f"**Iteration:** {progress.iteration}")
            st.markdown(f"**Leaves:** {progress.num_leaves}")
        with col2:
            st.markdown(f"**Parameters:** {progress.params}")
            st.markdown(f"**Current Loss:** {format_loss_value(progress.current_val_loss)}")
            st.markdown(f"**Best Loss:** {format_loss_value(progress.best_val_loss)}")

        if progress.separability_found:
            st.success("✅ Separability found!")

        # Show latest output snippet
        if st.session_state.sr_runner:
            output_lines = st.session_state.sr_runner.get_output_lines(max_lines=5)
            if output_lines:
                with st.expander("📜 Latest Output (last 5 lines)", expanded=True):
                    st.code("\n".join(output_lines[-5:]), language="text")

    elif progress.stage == "stageB":
        # Live indicator
        st.markdown("### 🔴 LIVE: Stage B - Analytic Refinement")

        # Show current model prominently
        if progress.model_expression:
            st.markdown(f"### Current Model: `{progress.model_expression}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stage:** Refining expression")
            st.markdown(f"**Parameters:** {progress.params}")
        with col2:
            st.markdown(f"**Current Loss:** {format_loss_value(progress.current_val_loss)}")
            st.markdown(f"**Best Loss:** {format_loss_value(progress.best_val_loss)}")

        # Show latest output snippet
        if st.session_state.sr_runner:
            output_lines = st.session_state.sr_runner.get_output_lines(max_lines=5)
            if output_lines:
                with st.expander("📜 Latest Output (last 5 lines)", expanded=True):
                    st.code("\n".join(output_lines[-5:]), language="text")

    elif progress.stage == "complete":
        st.success("✅ Symbolic regression complete!")
        st.markdown(f"**Total time:** {progress.elapsed_hours * 60:.2f} minutes")
        st.markdown(f"**Final loss:** {format_loss_value(progress.best_val_loss)}")

        if progress.final_expression:
            st.markdown("### Final Expression")
            st.code(progress.final_expression, language="python")

with tab2:
    st.subheader("Progress Details")

    # Model information
    st.markdown("### Current Model")
    if progress.model_expression:
        st.code(progress.model_expression, language="text")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iteration", progress.iteration)
        if progress.max_trial > 0:
            st.metric("Trial", f"{progress.trial}/{progress.max_trial}")
        st.metric("Leaves", progress.num_leaves)
    with col2:
        st.metric("Parameters", progress.params)
        if progress.dual_layer:
            st.info("🔄 Dual Layer Mode")
    with col3:
        st.metric("Current Loss", format_loss_value(progress.current_val_loss))
        st.metric("Best Loss", format_loss_value(progress.best_val_loss))

    # Status indicators
    st.markdown("### Status")
    status_cols = st.columns(3)
    with status_cols[0]:
        if progress.separability_found:
            st.success("✅ Separability Found")
        else:
            st.info("🔍 Searching...")
    with status_cols[1]:
        if progress.stageB_active:
            st.success("🎯 Stage B Active")
        else:
            st.info("Stage B Pending")
    with status_cols[2]:
        if progress.final_expression:
            st.success("✅ Complete!")
        else:
            st.info("⏳ Running...")

with tab3:
    st.subheader("Output Log")

    if st.session_state.sr_runner:
        output_lines = st.session_state.sr_runner.get_output_lines(max_lines=1000)

        if output_lines:
            # Show status
            if st.session_state.sr_running:
                st.info(f"🔄 Live output - {len(output_lines)} lines captured")
            else:
                st.success(f"✅ Complete - {len(output_lines)} total lines")

            # Show output in code block for better formatting
            output_text = "\n".join(output_lines)
            st.code(output_text, language="text", line_numbers=False)

            # Download button
            col1, col2 = st.columns([1, 3])
            with col1:
                # Create safe filename from model expression
                if progress.model_expression:
                    safe_name = (
                        progress.model_expression.replace(" ", "_")
                        .replace("/", "_")
                        .replace("(", "")
                        .replace(")", "")[:30]
                    )
                else:
                    safe_name = "output"

                st.download_button(
                    label="📥 Download Log",
                    data=output_text,
                    file_name=f"sr_{safe_name}.log",
                    mime="text/plain",
                )
            with col2:
                if st.session_state.sr_running:
                    st.caption(
                        f"Last line: {output_lines[-1][:80]}..."
                        if len(output_lines[-1]) > 80
                        else output_lines[-1]
                    )
        else:
            if st.session_state.sr_running:
                st.info("🔄 Starting... waiting for output...")
            else:
                st.info("No output yet. Start SR to see the log.")
    else:
        st.info("No SR process running. Click Start to begin.")

# Status bar at bottom
st.markdown("---")

if st.session_state.sr_running:
    # Get output info
    output_count = 0
    latest_line = "Starting..."
    if st.session_state.sr_runner:
        output_lines = st.session_state.sr_runner.get_output_lines(max_lines=1000)
        output_count = len(output_lines)
        if output_lines:
            latest_line = output_lines[-1][:100]

    # Show running status with details
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(
            f"🔄 **Running** ({progress.stage}) | {output_count} output lines | Updating every second..."
        )
    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    # Show latest line
    if output_count > 0:
        st.caption(f"**Latest:** {latest_line}")

elif progress.stage == "complete":
    st.success(f"✅ **Completed!** Final expression: {progress.final_expression or 'See above'}")
else:
    st.caption("Ready to start symbolic regression. Click ▶️ Start in the sidebar.")

# Auto-refresh while running
if st.session_state.sr_running:
    time.sleep(1)  # Refresh every second
    st.rerun()
