"""
Strategy Backtesting Tab Module - UPDATED WITH GITHUB WORKFLOW TRIGGER
Handles the backtesting UI and visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path

# Import Supabase client
from supabase import create_client, Client


# ============================================================================
# GITHUB WORKFLOW TRIGGER - NEW ADDITION
# ============================================================================

def run_backtest_via_github(strategy_id: int, strategy_config: dict):
    """
    Trigger backtest via GitHub Actions workflow
    """
    import os
    import sys
    
    # Check if GitHub token is available (try multiple env var names)
    github_token = None
    try:
        github_token = (
            os.environ.get("GITHUB_TOKEN") or 
            os.environ.get("G_TOKEN") or
            os.environ.get("GH_TOKEN") or
            st.secrets.get("G_TOKEN") or
            st.secrets.get("GITHUB_TOKEN") or
            st.secrets.get("GH_TOKEN")
        )
    except:
        pass
    
    github_owner = None
    try:
        github_owner = (
            os.environ.get("GITHUB_REPO_OWNER") or
            st.secrets.get("GITHUB_REPO_OWNER", "")
        )
    except:
        pass
    
    if not github_token:
        st.warning("⚠️ GitHub token not configured. Running locally...")
        run_backtest_local(strategy_id, strategy_config)
        return
    
    if not github_owner:
        st.error("❌ GITHUB_REPO_OWNER not configured.")
        st.info("Add GITHUB_REPO_OWNER to Streamlit secrets (your GitHub username)")
        return
    
    try:
        # Set environment variables for the trigger
        os.environ["GITHUB_TOKEN"] = github_token
        os.environ["GITHUB_REPO_OWNER"] = github_owner
        
        # Import the GitHub trigger module
        import requests
        
        repo_name = os.environ.get("GITHUB_REPO_NAME", "tradingview-analysis")
        api_base = f"https://api.github.com/repos/{github_owner}/{repo_name}"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        workflow_file = "backtest.yml"
        url = f"{api_base}/actions/workflows/{workflow_file}/dispatches"
        
        payload = {
            "ref": "main",
            "inputs": {
                "strategy_id": str(strategy_id),
                "verbose": "true"
            }
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 204:
            st.success(f"✅ Backtest workflow triggered via GitHub Actions!")
            st.info(f"🔄 Strategy ID: {strategy_id} is now running in GitHub Actions")
            
            # Show recent runs
            with st.expander("📊 Recent Workflow Runs", expanded=True):
                try:
                    runs_url = f"{api_base}/actions/workflows/{workflow_file}/runs"
                    runs_response = requests.get(runs_url, params={"per_page": 5}, headers=headers, timeout=10)
                    
                    if runs_response.status_code == 200:
                        runs = runs_response.json().get("workflow_runs", [])
                        if runs:
                            for run in runs[:5]:
                                status_emoji = {
                                    "completed": "✅",
                                    "in_progress": "🔄",
                                    "queued": "⏳",
                                    "failed": "❌"
                                }.get(run.get("status"), "❓")
                                
                                conclusion_emoji = {
                                    "success": "✅",
                                    "failure": "❌",
                                    "cancelled": "🚫"
                                }.get(run.get("conclusion", ""), "")
                                
                                st.markdown(
                                    f"{status_emoji} {conclusion_emoji} **Run #{run.get('run_number')}** - "
                                    f"{run.get('status')} - "
                                    f"{run.get('created_at', '')[:19]}"
                                )
                                st.markdown(f"[View Run]({run.get('html_url')})")
                                st.divider()
                        else:
                            st.info("No recent runs found")
                except Exception as e:
                    st.warning(f"Could not fetch workflow runs: {e}")
            
            st.info("⏱️ Refresh this page in 2-10 minutes to see results in 'View Results' tab.")
        else:
            st.error(f"❌ Failed to trigger workflow: {response.status_code}")
            st.error(f"Response: {response.text}")
            st.warning("Falling back to local execution...")
            run_backtest_local(strategy_id, strategy_config)
            
    except Exception as e:
        st.error(f"❌ Error triggering workflow: {e}")
        st.warning("Falling back to local execution...")
        run_backtest_local(strategy_id, strategy_config)


def run_backtest_local(strategy_id: int, strategy_config: dict):
    """
    Fallback: Run backtest locally using threading
    """
    import threading
    
    st.warning("🔄 Running backtest locally (GitHub Actions unavailable)")
    
    # Create and start background thread
    thread = threading.Thread(
        target=run_backtest_in_thread,
        args=(strategy_id, strategy_config),
        daemon=True
    )
    thread.start()
    
    st.success(f"✅ Backtest started locally! Strategy ID: {strategy_id}")
    st.info("🔄 The backtest is running in the background. Refresh this page in a few moments to see results.")
    st.info("⏱️ Depending on the date range, this may take 1-10 minutes.")
    
    # Show progress instructions
    with st.expander("📊 How to monitor progress"):
        st.write("""
        The backtest is now running. To see results:
        
        1. **Wait** 1-5 minutes (depends on date range)
        2. **Click 'Refresh'** button or reload page
        3. **Check 'View Results'** tab
        4. Look for strategy status changing from 'running' to 'completed'
        
        **Status indicators:**
        - `pending` - Strategy created, not started
        - `running` - Currently backtesting (wait for completion)
        - `completed` - Done! Results available
        - `failed` - Error occurred (check logs)
        """)


# ============================================================================
# REST OF THE FILE CONTINUES AS BEFORE...
# (I'll include just the key parts that reference the run function)
# ============================================================================
