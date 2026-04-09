import base64
import os
import subprocess
import threading
import time

import requests

URL = os.getenv('AIRPUFF_SENSE_URL', 'http://192.168.31.240:5000/api/sense')

WIDTH = int(os.getenv('STREAM_WIDTH', '1280'))
HEIGHT = int(os.getenv('STREAM_HEIGHT', '720'))
FPS = int(os.getenv('CAM_FPS', '20'))
UPLOAD_FPS = float(os.getenv('UPLOAD_FPS', '8'))
MJPEG_QUALITY = int(os.getenv('MJPEG_QUALITY', '82'))
POST_TIMEOUT = float(os.getenv('POST_TIMEOUT', '1.8'))

latest_jpeg = None
latest_seq = 0
latest_ts = 0.0
buf_lock = threading.Lock()


def camera_worker():
    global latest_jpeg, latest_seq, latest_ts
    cmd = [
        'rpicam-vid',
        '-n',
        '--codec', 'mjpeg',
        '--quality', str(MJPEG_QUALITY),
        '--width', str(WIDTH),
        '--height', str(HEIGHT),
        '--framerate', str(FPS),
        '--timeout', '0',
        '-o', '-',
    ]

    while True:
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            stream = proc.stdout
            if stream is None:
                raise RuntimeError('missing stdout')

            data = bytearray()
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                data.extend(chunk)

                while True:
                    soi = data.find(b'\xff\xd8')
                    if soi < 0:
                        if len(data) > 3_000_000:
                            del data[:-8192]
                        break
                    eoi = data.find(b'\xff\xd9', soi + 2)
                    if eoi < 0:
                        if soi > 0:
                            del data[:soi]
                        break

                    jpg = bytes(data[soi : eoi + 2])
                    del data[: eoi + 2]

                    with buf_lock:
                        latest_jpeg = jpg
                        latest_seq += 1
                        latest_ts = time.time()

            raise RuntimeError('camera stream closed')

        except Exception:
            pass
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass

        time.sleep(0.25)


def main():
    threading.Thread(target=camera_worker, daemon=True).start()

    sent = 0
    dropped = 0
    last_sent_seq = 0
    last_log = time.time()
    interval = 1.0 / max(UPLOAD_FPS, 0.5)

    while True:
        loop_start = time.time()

        seq = 0
        cap_ts = 0.0
        jpg = None
        with buf_lock:
            if latest_jpeg is not None:
                jpg = latest_jpeg
                seq = latest_seq
                cap_ts = latest_ts

        if jpg is None or seq == last_sent_seq:
            time.sleep(0.005)
            continue

        if last_sent_seq and seq - last_sent_seq > 1:
            dropped += seq - last_sent_seq - 1
        last_sent_seq = seq

        try:
            img_b64 = base64.b64encode(jpg).decode('ascii')
            requests.post(URL, json={'image': img_b64, 'audio': '', 'text': ''}, timeout=POST_TIMEOUT)
            sent += 1

            now = time.time()
            if now - last_log >= 2.0:
                lag_ms = (now - cap_ts) * 1000 if cap_ts else -1
                print(
                    f'sent={sent} drop={dropped} lag_ms={lag_ms:.0f} '
                    f'res={WIDTH}x{HEIGHT}@{FPS} upl={UPLOAD_FPS:.1f} b64={len(img_b64)}',
                    flush=True,
                )
                last_log = now

            dt = now - loop_start
            if dt < interval:
                time.sleep(interval - dt)

        except Exception:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
