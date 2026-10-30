"""Isolated LivePortrait subprocess. Never opens the application database."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services.multiview_enrollment_service import MultiViewEnrollmentService


def main():
    folder, repository = map(Path, sys.argv[1:3])
    manifest = json.loads((folder / "inputs.json").read_text())
    generator = MultiViewEnrollmentService(repository)
    results = []
    try:
        for row in manifest:
            frame = cv2.imread(str(folder / row["file"]))
            for image, info in generator.generate(frame, np.asarray(row["landmarks"], np.float32)):
                name = f'{row["source"]}_{int(info["target_yaw"])}.png'
                if not cv2.imwrite(str(folder / name), image):
                    raise RuntimeError("Unable to save generated view")
                results.append(dict(source=row["source"], file=name, **info))
        (folder / "outputs.json").write_text(json.dumps(results))
    finally:
        generator.stop()


if __name__ == "__main__":
    main()
