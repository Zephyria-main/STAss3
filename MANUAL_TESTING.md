# Manual Testing Evidence

**Project:** Macroinvertebrate Image Analysis System  
**Unit:** Software Technology 1 (4483/8995) — Assignment 3

---

## Test Scenarios

| # | Scenario | Input | Expected Result | Actual Result | Evidence |
|---|---|---|---|---|---|
| 1 | Missing dataset folder | `data/raw/` is empty | Summary prints warning, no crash | Passed | Screenshot 1 |
| 2 | Invalid image path in prediction | `not_a_file.jpg` | `[!] File not found` message shown | Passed | Screenshot 2 |
| 3 | Predict before model is trained | No `.joblib` file present | `FileNotFoundError` with helpful message | Passed | Screenshot 3 |
| 4 | Invalid menu option (console) | Enter `9` at main menu | "Invalid option" message, menu re-displayed | Passed | Screenshot 4 |
| 5 | Blank menu input | Press Enter with no input | "Invalid option" message, no crash | Passed | Screenshot 5 |
| 6 | Unsupported file extension | `file.txt` entered as image path | File not found / path exists but unreadable | Passed | Screenshot 6 |
| 7 | Valid end-to-end pipeline | Real dataset in `data/raw/` | Charts saved, model trained, prediction shown | Passed | Screenshot 7 |
| 8 | GUI choose image but no predict | Load image, close without predicting | No crash, state is preserved | Passed | Screenshot 8 |
| 9 | Predict without choosing image (GUI) | Click Predict with no file | Warning dialog "No Image Selected" shown | Passed | Screenshot 9 |
| 10 | Re-train without clearing model | Train twice in same session | Second model overwrites first, no error | Passed | Screenshot 10 |

---

## Notes

- All tests were run on the local development machine.
- Screenshots were captured during live runs.
- The dataset used for test 7 is the Kaggle Stream Macroinvertebrates dataset.
- No automated test framework was used; testing was manual and scenario-based as described above.
