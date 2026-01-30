name: Daily Stock Analysis

on:
  # Run daily at 6:00 AM EST (after market close)
  schedule:
    - cron: '0 11 * * *'  # 11:00 UTC = 6:00 AM EST
  
  # Allow manual triggering
  workflow_dispatch:
    inputs:
      run_detect:
        description: 'Run event detection'
        required: false
        default: 'true'
      run_collect:
        description: 'Run data collection'
        required: false
        default: 'true'
      run_analyze:
        description: 'Run analysis'
        required: false
        default: 'true'

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Create directories
        run: |
          mkdir -p credentials
          mkdir -p data/cache
          mkdir -p data/exports
          mkdir -p logs
      
      - name: Setup Google Sheets credentials
        env:
          GOOGLE_SHEETS_CREDENTIALS: ${{ secrets.GOOGLE_SHEETS_CREDENTIALS }}
        run: |
          # IMPORTANT: This file is created at runtime and never committed to git
          echo "$GOOGLE_SHEETS_CREDENTIALS" > credentials/google_sheets_credentials.json
          # Verify the file was created (without printing contents)
          if [ -f credentials/google_sheets_credentials.json ]; then
            echo "✓ Credentials file created successfully"
            # Verify it's valid JSON
            python -c "import json; json.load(open('credentials/google_sheets_credentials.json'))" && echo "✓ Valid JSON format"
          else
            echo "✗ Failed to create credentials file"
            exit 1
          fi
      
      - name: Create config file
        run: |
          cp config.example.yaml config.yaml
          echo "✓ Config file created from example"
      
      - name: Run analysis pipeline
        env:
          TRADINGVIEW_COOKIE: ${{ secrets.TRADINGVIEW_COOKIE }}
        run: |
          if [ "${{ github.event.inputs.run_detect }}" != "false" ] && \
             [ "${{ github.event.inputs.run_collect }}" != "false" ] && \
             [ "${{ github.event.inputs.run_analyze }}" != "false" ]; then
            # Run complete pipeline
            echo "Running complete pipeline..."
            python main.py --all --verbose
          else
            # Run selected steps
            ARGS=""
            [ "${{ github.event.inputs.run_detect }}" == "true" ] && ARGS="$ARGS --detect"
            [ "${{ github.event.inputs.run_collect }}" == "true" ] && ARGS="$ARGS --collect"
            [ "${{ github.event.inputs.run_analyze }}" == "true" ] && ARGS="$ARGS --analyze"
            echo "Running selected steps: $ARGS"
            python main.py $ARGS --verbose
          fi
      
      - name: Clean up sensitive files
        if: always()
        run: |
          # Remove credentials file to ensure it's not in any artifacts
          rm -f credentials/google_sheets_credentials.json
          echo "✓ Credentials cleaned up"
      
      - name: Upload logs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: analysis-logs
          path: logs/
          retention-days: 30
      
      - name: Upload exports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: data-exports
          path: data/exports/
          retention-days: 30
      
      - name: Notify on failure
        if: failure()
        run: |
          echo "Analysis pipeline failed. Check logs for details."
          # Add notification logic here (Slack, email, etc.)
