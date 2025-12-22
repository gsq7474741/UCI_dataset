from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ._base import BaseEnoseDataset, SampleRecord
from ._info import get_dataset_info
from ._utils import download_and_extract


class TwinGasSensorArrays(BaseEnoseDataset):
    """Twin gas sensor arrays (UCI Machine Learning Repository, id=361).

    Source:
        https://archive.ics.uci.edu/dataset/361/twin+gas+sensor+arrays

    Dataset summary (from the UCI page):
        5 replicates of an 8-MOX gas sensor array were exposed to different gas conditions (4 volatiles at
        10 concentration levels each).

    Dataset Information (UCI page):
        This dataset includes the recordings of five replicates of an 8-sensor array. Each unit holds 8 MOX
        sensors and integrates custom-designed electronics for sensor operating temperature control and signal
        acquisition. The same experimental protocol was followed to measure the response of the 5 twin units.
        Each day, a different unit was tested, which included the presentation of 40 different gas conditions,
        presented in random order. In particular, the unit under test was exposed to 10 concentration levels of
        Ethanol, Methane, Ethylene, and Carbon Monoxide. The duration of each experiment was 600 s, and the
        conductivity of each sensor was acquired at 100Hz.

        Channel, sensor type (from Figaro), and mean voltage in the heater are as follows:
        0: TGS2611 5.65 V
        1: TGS2612 5.65 V
        2: TGS2610 5.65 V
        3: TGS2602 5.65 V
        4: TGS2611 5.00 V
        5: TGS2612 5.00 V
        6: TGS2610 5.00 V
        7: TGS2602 5.00 V

        Presented concentration levels are as follows (in ppm):
        Ethylene: 12.5, 25, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0, 112.5, 125.0
        Ethanol: 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0, 112.5, 125.0
        Carbon Monoxide: 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0
        Methane: 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0

        Days in which each detection platform was tested.
        Unit 1: 4,10,15,21
        Unit 2: 1,7,11,16
        Unit 3: 2,8,14,17
        Unit 4: 3,9
        Unit 5: 18,22

        More information at:
        J. Fonollosa, L. Fernandez, A. Gutierrez-Galvez, R. Huerta, S. Marco.
        'Calibration transfer and drift counteraction in chemical sensor arrays using Direct Standardization'.
        Sensors and Actuators B: Chemical (2016).
        http://dx.doi.org/10.1016/j.snb.2016.05.089

        The data set can be used exclusively for research purposes. Commercial purposes are fully excluded.

    Variable Information (UCI page):
        The responses of the sensors are provided in a .txt file for each experiment. File name codes the unit
        number, gas (Ea: Ethanol, CO: CO, Ey: Ethylene, Me: Methane), concentration (010-100 of the
        corresponding gas), and repetition. For example, B1_GEa_F040_R2.txt indicates B1 (board 1), Ea
        (Ethanol), 50 ppm, Repetition 2. Each file includes the elapsed time (in seconds) and the resistance of
        each sensor (in KOhm). First column is time, and 8 following columns are channels 0-7 as specified
        before.

    DOI:
        https://doi.org/10.24432/C5MW3K
    """
    name = "twin_gas_sensor_arrays"

    gas_to_idx = {"Ea": 0, "CO": 1, "Ey": 2, "Me": 3}
    gas_ppm_factor = {"Ea": 1.25, "CO": 2.5, "Ey": 1.25, "Me": 2.5}

    _re = re.compile(r"^B(?P<board>\d+)_G(?P<gas>\w+)_F(?P<ppm>\d+)_R(?P<repeat>\d+)\.txt$")

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Optional[str] = None,
        download: bool = False,
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root,
            split=split,
            download=download,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

    def download(self) -> None:
        info = get_dataset_info(self.name)
        download_and_extract(info, self.dataset_dir, force=False, verify=True)

    def _check_exists(self) -> bool:
        # Twin 数据集 raw/ 下会整理到 data1/*.txt
        raw = self.raw_dir
        if not raw.exists():
            return False
        txt = list(raw.glob("*.txt"))
        if txt:
            return True
        if (raw / "data1").exists() and any((raw / "data1").glob("*.txt")):
            return True
        return False

    def _txt_dir(self) -> Path:
        d1 = self.raw_dir / "data1"
        if d1.exists():
            return d1
        return self.raw_dir

    def _make_dataset(self) -> List[SampleRecord]:
        txt_dir = self._txt_dir()
        files = sorted(txt_dir.glob("*.txt"))

        samples: List[SampleRecord] = []
        for p in files:
            m = self._re.match(p.name)
            if not m:
                continue

            gas = m.group("gas")
            if gas not in self.gas_to_idx:
                continue

            board = int(m.group("board"))
            ppm_code = int(m.group("ppm"))
            repeat = int(m.group("repeat"))

            ppm_value = ppm_code * self.gas_ppm_factor.get(gas, 1.0)

            target = {
                "gas": self.gas_to_idx[gas],
                "ppm": float(ppm_value),
                "board": board,
                "repeat": repeat,
            }

            meta: Dict[str, Any] = {"gas": gas, "ppm": float(ppm_value), "board": board, "repeat": repeat}
            samples.append(SampleRecord(sample_id=p.stem, path=p, target=target, meta=meta))

        # split: 目前不强制，留给下游/你后续加 splits.json
        if self.split is not None:
            raise ValueError("TwinGasSensorArrays 暂不支持 split（未提供 splits.json），请传 split=None")

        return samples

    def _load_sample(self, record: SampleRecord) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # 原始文件：time + 8 sensor columns
        df = pd.read_csv(record.path, sep=r"\s+", header=None)
        if df.shape[1] < 9:
            raise RuntimeError(f"Unexpected format: {record.path}")

        df = df.iloc[:, :9]
        df.columns = ["t_s"] + [f"sensor_{i}" for i in range(8)]
        return df, dict(record.target)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        parts = [base] if base else []
        parts.append("target={'gas': int, 'ppm': float, 'board': int, 'repeat': int}")
        return "\n".join(parts)
