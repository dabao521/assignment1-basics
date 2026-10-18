import argparse
from building_blocks.adamW_optimizer import AdamWOptimizer
from building_blocks.transformer_lm import TransformerLM
from building_blocks.common import get_lr_cosine_schedule, get_batch, cross_entropy_loss, gradient_clipping, save_checkpoint, load_checkoint
import numpy as np
import numpy.typing as npt
import torch
import csv
import time

def main() -> None:
    parser = argparse.ArgumentParser(description="CS336 Transformer Training")

    # datasets
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)

    # model hyperparameters
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=352)
    parser.add_argument("--theta", type=float, default=1000.0)

    # training hyperparameters
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10000)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)


    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-file", type=str, default="training_log.csv")
    
    parser.add_argument("--overfit-one-batch", action="store_true")

    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-file", type=str, default=None)

    args = parser.parse_args()
    print(f"input hyper parameters: {args}")

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.theta,
        args.device,
    )

    optimizer = AdamWOptimizer(
        model.parameters(),
        args.learning_rate,
        (args.beta1, args.beta2),
        1e-8,
        args.weight_decay,
    )
    train_data: npt.NDArray = np.load(
        args.train_data,
        mmap_mode="r",
    )
    val_data: npt.NDArray = np.load(
        args.val_data,
        mmap_mode="r",
    )

    fixed_batch = None
    if args.overfit_one_batch:
        fixed_batch = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

    train_losses: list[tuple[int,float]] = []
    val_losses: list[tuple[int,float]] = []

    start_iteration:int = 0
    if args.resume_from is not None:
        start_iteration = load_checkoint(
            args.resume_from,
            model,
            optimizer,
        )

    if args.device == "mps":
            torch.mps.synchronize()
    start_time = time.perf_counter()

    for iteration in range(start_iteration, args.num_iterations):
        # 1. update learning rate
        lr = get_lr_cosine_schedule(
            iteration,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. sample training batch
        if args.overfit_one_batch:
            inputs,targets = fixed_batch
        else:
            inputs, targets = get_batch(
                train_data,
                args.batch_size,
                args.context_length,
                args.device,
            )

        # 3. forward pass
        logits = model(inputs)

        # 4. compute the cross-entropy loss
        loss = cross_entropy_loss(logits, targets)

        # 5. clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # 6. loss.backward()
        loss.backward()

        # 7. gradient clipping
        gradient_clipping(model.parameters(), args.max_l2_norm)

        # 8  optimizer.step
        optimizer.step()

        # 9. periodic logging / validation / checkpointing
        step = iteration + 1
        if step % args.log_interval == 0:
            if args.device == "mps":
                torch.mps.synchronize()
            elapsed_seconds = time.perf_counter() - start_time
            print(
                f"step={iteration+1}, "
                f"train_loss={loss.item():.4f}, "
                f"lr={lr:6f}"
            )
            train_losses.append((step, elapsed_seconds, loss.item()))

        if (
                not args.overfit_one_batch
                and step % args.eval_interval == 0
            ):
            model.eval()

            losses: list[float] = []

            with torch.no_grad():
                for _ in range(args.eval_batches):
                    val_inputs, val_targets = get_batch(
                        val_data,
                        args.batch_size,
                        args.context_length,
                        args.device,
                    )
                    val_logits = model(val_inputs)
                    val_loss = cross_entropy_loss(val_logits, val_targets)
                    losses.append(val_loss.item())

            model.train()
            val_loss = sum(losses) / len(losses)
            if args.device == "mps":
                torch.mps.synchronize()
            elapsed_seconds = time.perf_counter() - start_time
            val_losses.append((step, elapsed_seconds, val_loss))
            print(f"step={iteration + 1}, val_loss={val_loss:.4f}")

        if (
            args.checkpoint_file is not None
            and step % args.checkpoint_interval == 0
            ):
            save_checkpoint(
                model,
                optimizer,
                step,
                args.checkpoint_file,
            )

    # save checkpoint one more time
    if args.checkpoint_file is not None:
        save_checkpoint(
            model,
            optimizer,
            args.num_iterations,
            args.checkpoint_file,
        )

    hyperparams: dict[str, str|int|float|bool] = vars(args)
    
    with open(args.log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "elapsed_seconds", "type", "loss", *hyperparams.keys()])

        for step, elapsed, loss in train_losses:
            writer.writerow([step, elapsed, "train", loss, *hyperparams.values()])

        for step, loss in val_losses:
            writer.writerow([step, elapsed, "val", loss, *hyperparams.values()])

if __name__ == "__main__":
    main()
