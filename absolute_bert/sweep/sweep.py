import logging

import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import absolute_bert.models.absolute_bert.models as abs_bert
from absolute_bert import loss
from absolute_bert.base_types import (
    EncoderInputs,
    EncoderInputsWithLabels,
    LanguageModel,
)
from absolute_bert.bi_encoder import EncoderEmbeddingPool, EncoderOutputPool
from absolute_bert.extractor import ModuleParamStatsExtractor
from absolute_bert.loggers import WandbLogger

from . import setup
from .evaluate import BeirBenchmark
from .optimizer import make_adamw
from .scheduler import make_scheduler
from .train import MultiLossAverager, format_losses
from .utils import log_step

# logging.getLogger("beir").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"using device `{device}` for sweeping")

config = setup.get_config()

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

artifact = wandb.use_artifact("ghuang-nlp/uncategorized/wikipedia.en-0.01:v0", type="dataset")
artifact_dir = artifact.download()
datadict = load_from_disk(artifact_dir)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=config.train.masking_probability
)
train_loader = DataLoader(
    datadict["train"], batch_size=config.batch_sizes.train, collate_fn=collator
)
val_loader = DataLoader(datadict["test"], batch_size=config.batch_sizes.val, collate_fn=collator)

# %% preparing train process

model: LanguageModel = abs_bert.AbsoluteBertLM(config).to(device)
loss_fn = loss.CrossEntropy(model)
logging.info(f"using model `{repr(model)}`")


optimizer = make_adamw(model, config=config.optimizer)
scheduler = make_scheduler(optimizer, num_batches=len(train_loader), config=config.scheduler)
averager = MultiLossAverager()

# %% evaluation setup
static_embedding_encoder = EncoderEmbeddingPool.from_model(model, tokenizer, device)
model_output_encoder = EncoderOutputPool.from_model(model, tokenizer, device)
benchmark = BeirBenchmark(corpus_name="scifact")
param_extractor = ModuleParamStatsExtractor(model, config.logging.params.rules)
logging_keys = list(param_extractor.extract_stats().keys())
logging.info(f"logging following params in model: {repr(logging_keys)}")

wandb_logger = WandbLogger()

def evaluate_and_log(tag: str, epoch_num: int | None = None):
    with log_step(step=global_step, tag=tag):
        output_metrics = benchmark.run(model_output_encoder, config.batch_sizes.ir)
        static_embeddings_metrics = benchmark.run(static_embedding_encoder, config.batch_sizes.ir)
        metrics = [
            ("model_output", output_metrics),
            ("static_embeddings", static_embeddings_metrics),
        ]
        wandb_logger.log_beir_metrics_without_commit(metrics, global_step, epoch_num)




# %% train start

global_step = 0

model.eval()


evaluate_and_log("global_step 0")

for epoch_num in range(config.train.num_epochs):

    bar = tqdm(train_loader)
    for _batch_num, batch in enumerate(bar):

        model.train()
        optimizer.zero_grad()
        inputs = EncoderInputs.from_mapping(batch.to(device))
        predicts, labels = model(inputs)
        loss = loss_fn(predicts, labels)
        final_loss, loss_dict = format_losses(loss, clip_value=config.train.clip_loss)

        if global_step % config.logging.train.every_n_steps == 0:
            wandb.log({f"train/{k}": v for k, v in loss_dict.items()}, step=global_step)

        bar.set_postfix(loss_dict)

        # with torch.autograd.detect_anomaly(True):
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)

        final_loss.backward()
        optimizer.step()
        global_step += 1
        scheduler.step()

        if global_step % config.logging.val.every_n_steps == 0:
            with log_step(step=global_step, tag="val"):
                model.eval()

                with torch.no_grad():
                    val_bar = tqdm(val_loader, leave=False)
                    for batch in val_bar:
                        inputs = EncoderInputsWithLabels.from_mapping(batch.to(device))
                        predicts, labels = model(inputs)
                        loss = loss_fn(predicts, labels)
                        _, loss_dict = format_losses(loss, clip_value=config.train.clip_loss)

                        batch_size = labels.size(0)
                        averager.update(loss_dict, batch_size)

                        val_bar.set_postfix(loss_dict)

                    avg_losses = averager.compute()

                wandb_logger.log_dict_without_commit(
                    {f"val/{k}": v for k, v in avg_losses.items()},
                    global_step,
                )
                averager.reset()

        if global_step % config.logging.params.every_n_steps == 0:
            with log_step(step=global_step, tag="params"):
                model.eval()
                param_stats = param_extractor.extract_stats()
                wandb_logger.log_param_stats_without_commit(
                    param_stats,
                    global_step,
                )

        if global_step % config.logging.ir.every_n_steps == 0:
            model.eval()
            evaluate_and_log("beir")

        # del batch, loss

        if (config.train.max_steps > 0) and (global_step > config.train.max_steps):
            break

    model.eval()
    evaluate_and_log("beir", epoch_num)

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
