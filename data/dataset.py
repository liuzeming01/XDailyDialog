"""Dialogue dataset."""

import json
import datasets

logger = datasets.logging.get_logger(__name__)


class Dataset(datasets.GeneratorBasedBuilder):
    """Dialogue dataset."""
    def _info(self):
        features = datasets.Features({"src": datasets.Value("string"), "tgt": datasets.Value("string")})
        return datasets.DatasetInfo(description="Dialogue dataset.", features=features)

    def _split_generators(self, dl_manager):
        data_files = self.config.data_files

        splits = []
        if 'train' in data_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}))
        if 'validation' in data_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["validation"]}))
        if 'test' in data_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}))

        return splits

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {"src": data["src"], "tgt": data["tgt"]}
