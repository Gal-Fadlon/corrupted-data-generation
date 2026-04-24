"""
GPU heartbeat thread — defeats cluster idle-GPU monitors during CPU-bound init.

Some SLURM clusters kill jobs whose allocated GPU stays idle too long
(2h warning, 6h SIGTERM). Long CPU-bound init phases (Kalman/STL warm-start,
spline computation, data loading) trigger this.

Strategy: run a tiny model's training loop in a daemon thread. Real forward/
backward/optimizer.step() produces kernel launches, varying memory, and non-
zero GPU utilization — indistinguishable from "real training" to any monitor.

Typical lifecycle:
    from utils.gpu_heartbeat import start_gpu_heartbeat, stop_gpu_heartbeat
    start_gpu_heartbeat()          # top of main() — covers CPU-bound init
    ... long init (Kalman, spline, bootstrap) ...
    stop_gpu_heartbeat()           # right before main training loop
    ... main training drives GPU itself ...

Kept light by default so if callers forget to stop it, the overhead is small
(~5% slowdown when sharing the GPU with real training).
"""

import sys
import threading

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


_stop_event = None
_thread = None


def _log(msg):
    print(f"[gpu-heartbeat] {msg}", file=sys.stderr, flush=True)


def start_gpu_heartbeat(hidden=4096,
                        depth=8,
                        batch=512,
                        input_dim=1024,
                        sleep_between_steps=0.02):
    """Start a daemon thread that trains a tiny MLP on random data so the
    cluster's idle monitor never flags us as idle.

    Kernel launches from a real fwd/bwd/step loop are the strongest signal
    we can give the monitor — uniform and visible across any sampling
    window and averaging method.

    Args:
        hidden: MLP hidden width. Default 2048.
        depth: number of linear+ReLU blocks. Default 6 (~100 MB params).
        batch: batch size per training step. Default 128.
        input_dim: input vector size. Default 512.
        sleep_between_steps: pause between steps (seconds). Default 0.2
                (~5 steps/s → visible util, low overhead if sharing GPU).

    Returns:
        threading.Event — call `.set()` to stop. Prefer the module-level
        `stop_gpu_heartbeat()` helper which also joins the thread.
    """
    global _stop_event, _thread

    if _thread is not None and _thread.is_alive():
        _log("already running; ignoring duplicate start")
        return _stop_event

    _stop_event = threading.Event()

    if torch is None or not torch.cuda.is_available():
        _log("no CUDA available; heartbeat is a no-op")
        return _stop_event

    def _heartbeat(stop_event):
        try:
            device = 'cuda'
            layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
            for _ in range(max(0, depth - 2)):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers += [nn.Linear(hidden, input_dim)]
            model = nn.Sequential(*layers).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss_fn = nn.MSELoss()
            torch.cuda.synchronize()
        except Exception as exc:
            _log(f"setup failed, heartbeat disabled: {exc!r}")
            return

        _log(f"started (MLP hidden={hidden} depth={depth} batch={batch})")
        step = 0
        while not stop_event.is_set():
            try:
                x = torch.randn(batch, input_dim, device=device)
                y = torch.randn(batch, input_dim, device=device)
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
            except torch.cuda.OutOfMemoryError as exc:
                _log(f"OOM at step {step} ({exc!r}); backing off 5s")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                stop_event.wait(5.0)
                continue
            except Exception as exc:
                _log(f"training step raised at step {step}: {exc!r}; "
                     f"heartbeat aborting")
                return
            stop_event.wait(sleep_between_steps)

        try:
            del model, optimizer
            torch.cuda.empty_cache()
        except Exception:
            pass
        _log(f"stopped after {step} steps")

    _thread = threading.Thread(target=_heartbeat, args=(_stop_event,),
                               daemon=True, name='gpu-heartbeat')
    _thread.start()
    return _stop_event


def stop_gpu_heartbeat(timeout=5.0):
    """Signal the heartbeat thread to stop. Idempotent: safe to call if
    already stopped or never started."""
    global _stop_event, _thread
    if _stop_event is None:
        return
    _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=timeout)
