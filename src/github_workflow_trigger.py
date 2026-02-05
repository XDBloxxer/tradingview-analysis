#!/usr/bin/env python3
"""
GitHub Workflow Trigger
Triggers workflow_dispatch events via GitHub API
"""

import os
import requests
import logging
from typing import Optional


class GitHubWorkflowTrigger:
    """Trigger GitHub Actions workflows via API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Get from environment
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.repo_owner = os.environ.get("GITHUB_REPO_OWNER", "your-username")
        self.repo_name = os.environ.get("GITHUB_REPO_NAME", "tradingview-analysis")
        
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable required")
        
        self.api_base = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        self.headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    
    def trigger_backtest(
        self,
        strategy_id: int,
        verbose: bool = False
    ) -> bool:
        """
        Trigger backtest workflow for a strategy
        
        Args:
            strategy_id: Strategy ID from database
            verbose: Enable verbose logging
            
        Returns:
            True if triggered successfully
        """
        workflow_file = "backtest.yml"
        url = f"{self.api_base}/actions/workflows/{workflow_file}/dispatches"
        
        payload = {
            "ref": "main",  # or "master" - adjust to your default branch
            "inputs": {
                "strategy_id": str(strategy_id),
                "verbose": "true" if verbose else "false"
            }
        }
        
        try:
            self.logger.info(f"Triggering backtest workflow for strategy {strategy_id}...")
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code == 204:
                self.logger.info(f"✅ Workflow triggered successfully")
                return True
            else:
                self.logger.error(
                    f"❌ Failed to trigger workflow: {response.status_code} - {response.text}"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error triggering workflow: {e}")
            return False
    
    def get_workflow_runs(
        self,
        strategy_id: Optional[int] = None,
        limit: int = 10
    ) -> list:
        """
        Get recent workflow runs
        
        Args:
            strategy_id: Optional filter by strategy ID
            limit: Number of runs to return
            
        Returns:
            List of workflow run dictionaries
        """
        workflow_file = "backtest.yml"
        url = f"{self.api_base}/actions/workflows/{workflow_file}/runs"
        
        params = {"per_page": limit}
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                runs = response.json().get("workflow_runs", [])
                
                # Filter by strategy_id if provided
                if strategy_id:
                    runs = [
                        r for r in runs 
                        if r.get("name") == f"Strategy {strategy_id}"
                    ]
                
                return runs
            else:
                self.logger.error(f"Failed to get workflow runs: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting workflow runs: {e}")
            return []
