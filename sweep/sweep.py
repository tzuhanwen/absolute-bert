import logging

import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from absolute_bert.model import lm_registry
from absolute_bert.loss import CrossEntropy
from absolute_bert.base_types import (
    EncoderInputs,
    EncoderInputsWithLabels,
    LanguageModel,
)
from absolute_bert.bi_encoder import EncoderEmbeddingPool, EncoderOutputPool
from absolute_bert.extractor import ModuleParamStatsExtractor
from absolute_bert.loggers import WandbLogger
from absolute_bert.utils import init_logging
from absolute_bert.sweep import setup
from absolute_bert.sweep.evaluate import BeirBenchmark
from absolute_bert.sweep.extractors import get_absolute_bert_semantic_summary
from absolute_bert.sweep.optimizer import make_adamw
from absolute_bert.sweep.scheduler import make_scheduler
from absolute_bert.sweep.train import MultiLossAverager, format_losses
from absolute_bert.sweep.utils import log_step
from absolute_bert.tracker import UpdateRatioTracker

if __name__ == "__main__":
    init_logging()

# logging.getLogger("beir").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"using device `{device}` for sweeping")

config_unresolved = setup.get_config()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config = config_unresolved.resolve(tokenizer.vocab_size)
logger.info(f"config resolved: {config=}")

run = wandb.init(
    **config.wandb.to_dict(),
    config=config.to_dict(),
    save_code=True,
)

# if not cfg.testing:  # save_before_train
#     temp_dir = get_saving_dir(cfg, model, for_initial=True)
#     if temp_dir:
#         trainer.save_model(temp_dir)

# %% data preparation
logger.info("data preparation")

artifact = wandb.use_artifact("ghuang-nlp/dataset/wikipedia.en-0.01:v0", type="dataset")
artifact_dir = artifact.download()
datadict = load_from_disk(artifact_dir)


collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=config.train.masking_probability
)
train_loader = DataLoader(
    datadict["train"], batch_size=config.micro_batch_sizes.train, collate_fn=collator
)
val_loader = DataLoader(
    datadict["test"], batch_size=config.micro_batch_sizes.val, collate_fn=collator
)

# %% preparing train process
logger.info("preparing train process")

# load model and test
model: LanguageModel = lm_registry[config.train.model_type](config.model).to(device)
batch = next(iter(train_loader))
model.forward(EncoderInputs.from_mapping(batch.to(device)))
logger.info(f"using model `{repr(model)}`")


loss_fn = CrossEntropy(model, config.loss)
optimizer = make_adamw(model, config=config.optimizer)
update_ratio_tracker = UpdateRatioTracker(model, optimizer)
effective_num_batches = int(len(train_loader) / config.train.accum_steps)
scheduler = make_scheduler(
    optimizer, effective_num_batches=effective_num_batches, config=config.scheduler
)
averager = MultiLossAverager()

# %% evaluation setup
logger.info("evaluation setup")

static_embedding_encoder = EncoderEmbeddingPool.from_model(model.base_model, tokenizer, device)
model_output_encoder = EncoderOutputPool.from_model(model.base_model, tokenizer, device)

corpus_name = "scifact"
benchmark = BeirBenchmark(corpus_name=corpus_name)
param_extractor = ModuleParamStatsExtractor(model, config.logging.params.rules)
logging_keys = list(param_extractor.extract_stats().keys())
logging.info(f"logging following parameter stats on named modules: {repr(logging_keys)}")

wandb_logger = WandbLogger()


def run_benchmarks_and_log(tag: str, epoch_num: int | None = None):
    with log_step(step=global_step, tag=tag):
        output_metrics = benchmark.run(model_output_encoder, config.micro_batch_sizes.ir)
        static_embeddings_metrics = benchmark.run(
            static_embedding_encoder, config.micro_batch_sizes.ir
        )
        metrics = [
            ("model_output", output_metrics),
            ("static_embeddings", static_embeddings_metrics),
        ]
        wandb_logger.log_beir_metrics(metrics, global_step, corpus_name, epoch_num)


def log_params():
    with log_step(step=global_step, tag="params"):
        param_stats = param_extractor.extract_stats()
        wandb_logger.log_param_stats(
            param_stats,
            global_step,
        )


def log_semantic_summary():
    with log_step(step=global_step, tag="semantic_summary"):
        summary = get_absolute_bert_semantic_summary(model, tokenizer)
        wandb_logger.dump_strings(
            {f"{layer_name}.txt": semantic for layer_name, semantic in summary}, 
            "semantic_summary", 
            global_step
        )


# %% train start
logger.info("train start")

global_step = 0

model.eval()
log_params()
log_semantic_summary()
run_benchmarks_and_log("global_step 0")

for epoch_num in range(config.train.num_epochs):

    bar = tqdm(train_loader)
    for _micro_batch_num, micro_batch in enumerate(bar):

        model.train()

        optimizer.zero_grad()
        inputs = EncoderInputs.from_mapping(micro_batch.to(device))
        predicts, labels = model(inputs)
        loss = loss_fn(predicts, labels)
        final_loss, loss_dict = format_losses(loss, clip_value=config.train.clip_loss)

        bar.set_postfix(loss_dict)

        (final_loss / config.train.accum_steps).backward()
        global_step += 1

        if global_step % config.train.accum_steps == 0:
            if (
                global_step
                % (config.logging.train.every_n_effective_steps * config.train.accum_steps)
                == 0
            ):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                with update_ratio_tracker.track():
                    optimizer.step()
                wandb.log(
                    {f"train/{k}": v for k, v in loss_dict.items()}
                    | {
                        f"train/update_ratio": update_ratio_tracker.ratio,
                        "train/gradient_norm": gn,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                    commit=False,
                )

            else:
                optimizer.step()

            scheduler.step()

        model.eval()
        with torch.no_grad():
            if global_step % config.logging.val.every_n_steps == 0:
                with log_step(step=global_step, tag="val"):
                    val_bar = tqdm(val_loader, leave=False)
                    for batch in val_bar:
                        inputs = EncoderInputsWithLabels.from_mapping(batch.to(device))
                        predicts, labels = model(inputs)
                        loss = loss_fn(predicts, labels)
                        _, loss_dict = format_losses(loss, clip_value=config.train.clip_loss)
                        averager.update(loss_dict, batch_size=labels.size(0))

                        val_bar.set_postfix(loss_dict)

                    avg_losses = averager.compute()

                    wandb_logger.log_dict(
                        {f"val/{k}": v for k, v in avg_losses.items()},
                        global_step,
                    )
                averager.reset()

            if global_step % config.logging.params.every_n_steps == 0:
                log_params()

            if global_step % config.logging.ir.every_n_steps == 0:
                run_benchmarks_and_log("beir")

            if global_step % config.logging.semantic.every_n_steps == 0:
                log_semantic_summary()

            if (config.train.max_steps > 0) and (global_step > config.train.max_steps):
                break

    model.eval()
    with torch.no_grad():
        run_benchmarks_and_log("beir", epoch_num)

# model_artifact = wandb.Artifact(name="model",
#   type="model",
#   # description="trained with 2-1-training_with_msmarco",
#   metadata=training_args_config)
# model_artifact.add_dir(saving_dir)
# run.log_artifact(model_artifact)

# run.log_code(**get_code_files_aggregating_functions(cfg.env.project_root))

# if not cfg.testing:
#     if not os.path.exists(saving_dir):
#         os.mkdir(saving_dir)

#     trainer.save_model(saving_dir)
#     trainer.save_state()
#     torch.save(model, saving_dir/'pytorch.pt')

"""### for markdown recording"""

# for polaritical_docs in zip(test_cossim_before, test_cossim_after):  # pos & neg
#     print(f"||{'|'.join(str(idx) for idx in range(10))}|")
#     for timepoint, values in zip(['before', 'after'], polaritical_docs):  # before & after
#         print(f"|{timepoint}|{'|'.join(f'{val:.4f}' for val in values.tolist())}|")
#     print()

wandb.finish()
