#*******************************
#Author: u323115
#Assessment 3 
#Programming: u3231515
#*******************************
#
# main.py (project root)
# I keep this entry point at the project root so
# a marker or tutor sees first when they open the submission. 
#
# Usage:
#   python main.py               — runs the full Stage 1 + Stage 2 pipeline
#   python -m src.console_app   — launches the interactive console menu
#   python -m src.app           — launches the Tkinter GUI
#
# Unit tutorial / guidance — acknowledgement (Step 10: batch entry script):
# Based on: Assignment 3 Full Guidance, Step 10 (minimal main() constructing WorkflowService
# and calling run_full_pipeline). Weekly lab materials use the same “thin entry point” idea.
# How this project extends it: script lives at project root as main.py (not under src/) so
# markers can run `python main.py` per submission layout; behaviour delegates entirely to
# WorkflowService. See IMPLEMENTATION_SUMMARY.md.

from src.services.workflow_service import WorkflowService


def main() -> None:
    """Run the full non-interactive Stage 1 and Stage 2 pipeline.

    I designed this so the marker can run one command to confirm the
    entire project works end to end without clicking through menus.
    """
    workflow = WorkflowService()
    workflow.run_full_pipeline()


if __name__ == "__main__":
    main()
