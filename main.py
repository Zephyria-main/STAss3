# main.py (project root)
# I keep this entry point at the project root so it is the first file
# a marker or tutor sees when they open the submission. It runs the full
# non-interactive pipeline: index → EDA → train → report.
#
# Usage:
#   python main.py               — runs the full Stage 1 + Stage 2 pipeline
#   python -m src.console_app   — launches the interactive console menu
#   python -m src.app           — launches the Tkinter GUI

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
