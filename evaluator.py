"""ESC-50 evaluation helpers for zero-shot PANNs inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from inference import PannsPredictor


ESC50_TO_AUDIOSET: Dict[str, List[str]] = {
    "dog": ["Bark", "Dog", "Whimper", "Howl"],
    "rooster": ["Fowl", "Chicken, rooster", "Crow"],
    "pig": ["Pig", "Oink"],
    "cow": ["Moo", "Cattle, bovinae", "Livestock, farm animals, working animals"],
    "frog": ["Frog"],
    "cat": ["Cat", "Meow"],
    "hen": ["Chicken, rooster", "Fowl", "Cluck"],
    "insects": ["Insect", "Cricket", "Mosquito", "Fly, housefly"],
    "sheep": ["Sheep", "Bleat"],
    "crow": ["Crow", "Caw"],
    "rain": ["Rain", "Raindrop", "Thunderstorm"],
    "sea_waves": ["Ocean", "Waves, surf", "Water"],
    "crackling_fire": ["Crackle", "Fire"],
    "crickets": ["Cricket"],
    "chirping_birds": ["Bird vocalization, bird call, bird song", "Chirp, tweet"],
    "water_drops": ["Drip", "Water tap, faucet", "Raindrop"],
    "wind": ["Wind", "Rustling leaves"],
    "pouring_water": ["Pour", "Water tap, faucet"],
    "toilet_flush": ["Toilet flush", "Water"],
    "thunderstorm": ["Thunderstorm", "Thunder"],
    "crying_baby": ["Baby cry, infant cry"],
    "sneezing": ["Sneeze"],
    "clapping": ["Clapping"],
    "breathing": ["Breathing"],
    "coughing": ["Cough"],
    "footsteps": ["Walk, footsteps"],
    "laughing": ["Laughter"],
    "brushing_teeth": ["Toothbrush"],
    "snoring": ["Snoring"],
    "drinking_sipping": ["Drinking", "Gulp"],
    "door_wood_knock": ["Knock"],
    "mouse_click": ["Clicking", "Computer keyboard"],
    "keyboard_typing": ["Typing", "Computer keyboard"],
    "door_wood_creaks": ["Door", "Creak"],
    "can_opening": ["Canidae", "Crackle"],  # fallback gets corrected below if unavailable
    "washing_machine": ["Washing machine"],
    "vacuum_cleaner": ["Vacuum cleaner"],
    "clock_alarm": ["Alarm clock"],
    "clock_tick": ["Tick-tock"],
    "glass_breaking": ["Glass", "Chink, clink"],
    "helicopter": ["Helicopter"],
    "chainsaw": ["Chainsaw"],
    "siren": ["Siren"],
    "car_horn": ["Vehicle horn, car horn, honking"],
    "engine": ["Engine", "Idling"],
    "train": ["Train", "Rail transport"],
    "church_bells": ["Church bell"],
    "airplane": ["Aircraft", "Fixed-wing aircraft, airplane"],
    "fireworks": ["Fireworks"],
    "hand_saw": ["Sawing", "Tools"],
}

# Prefer more plausible labels if the first one does not exist in the loaded ontology.
ESC50_LABEL_FIXUPS = {
    "can_opening": ["Opening or closing", "Cutlery, silverware", "Crackle"],
}


@dataclass
class EvaluationSummary:
    """Aggregate metrics for ESC-50 evaluation."""

    top1_accuracy: float
    top5_accuracy: float
    sample_count: int
    confusion_matrix_path: Path


class ESC50Evaluator:
    """Run zero-shot evaluation by mapping AudioSet scores to ESC-50 labels."""

    def __init__(self, predictor: PannsPredictor) -> None:
        self.predictor = predictor
        self.available_labels = set(predictor.labels)
        self.esc50_classes = sorted(ESC50_TO_AUDIOSET.keys())
        self.mapping = self._build_mapping()

    def _build_mapping(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for esc_label, candidates in ESC50_TO_AUDIOSET.items():
            selected = [label for label in candidates if label in self.available_labels]
            if not selected and esc_label in ESC50_LABEL_FIXUPS:
                selected = [label for label in ESC50_LABEL_FIXUPS[esc_label] if label in self.available_labels]
            if not selected:
                selected = [candidate for candidate in candidates[:1]]
                logging.warning("No exact AudioSet mapping found for %s, using fallback label %s", esc_label, selected[0])
            mapping[esc_label] = selected
        return mapping

    def evaluate(
        self,
        data_root: str,
        test_fold: int = 1,
        subset_ratio: float = 1.0,
        output_path: str = "confusion_matrix.png",
    ) -> EvaluationSummary:
        root = Path(data_root).expanduser().resolve()
        meta_csv = root / "meta" / "esc50.csv"
        audio_dir = root / "audio"
        if not meta_csv.exists():
            raise FileNotFoundError(f"ESC-50 metadata not found: {meta_csv}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"ESC-50 audio directory not found: {audio_dir}")

        df = pd.read_csv(meta_csv)
        test_df = df[df["fold"] == test_fold].copy()
        if not 0 < subset_ratio <= 1:
            raise ValueError("subset_ratio must be in the range (0, 1].")
        if subset_ratio < 1.0:
            test_df = test_df.sample(frac=subset_ratio, random_state=42).sort_values("filename")

        y_true: List[str] = []
        y_pred: List[str] = []
        top5_hits = 0

        for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="Evaluating ESC-50"):
            audio_path = audio_dir / row.filename
            raw_scores = self.predictor.esc50_scores(str(audio_path))
            esc_scores = {
                esc_label: max(raw_scores.get(audioset_label, 0.0) for audioset_label in audioset_labels)
                for esc_label, audioset_labels in self.mapping.items()
            }
            ranked = sorted(esc_scores.items(), key=lambda item: item[1], reverse=True)
            top1_label = ranked[0][0]
            top5_labels = [label for label, _ in ranked[:5]]

            y_true.append(row.category)
            y_pred.append(top1_label)
            top5_hits += int(row.category in top5_labels)

        cm = confusion_matrix(y_true, y_pred, labels=self.esc50_classes)
        fig, ax = plt.subplots(figsize=(18, 18))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.esc50_classes).plot(
            ax=ax,
            xticks_rotation=90,
            colorbar=False,
        )
        plt.tight_layout()
        cm_path = Path(output_path).expanduser().resolve()
        fig.savefig(cm_path, dpi=200)
        plt.close(fig)

        top1 = float(np.mean(np.array(y_true) == np.array(y_pred)))
        top5 = top5_hits / len(y_true) if y_true else 0.0
        return EvaluationSummary(
            top1_accuracy=top1,
            top5_accuracy=top5,
            sample_count=len(y_true),
            confusion_matrix_path=cm_path,
        )
