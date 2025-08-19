from locust import HttpUser, task, constant, events
import time, json, random, requests

QUERIES = [
    "Apa saja fasilitas di Departemen Ilmu Komputer?",
    "Jelaskan KBK/Penjurusan di Ilkom UPI",
    "Visi dan Misi Pendidikan Ilmu Komputer",
    "Bagaimana proses pendaftaran mahasiswa baru?",
    "Daftar mata kuliah inti semester awal",
    "Apa itu Keluarga Mahasiswa Komputer?",
    "Informasi beasiswa untuk mahasiswa Ilkom",
    "Profil lulusan dan capaian pembelajaran",
]

TIMEOUT = 90

class ChatUser(HttpUser):
    wait_time = constant(0.5)

    @task
    def ask(self):
        q = random.choice(QUERIES)
        url = self.host + "/chat"
        payload = {"message": q, "history": []}

        start = time.perf_counter()
        ttft = None
        nbytes = 0
        exc = None
        status_code = None
        try:
            with requests.post(url, json=payload, stream=True, timeout=TIMEOUT) as r:
                status_code = r.status_code
                if status_code != 200:
                    exc = Exception(f"HTTP {status_code}")
                else:
                    for raw in r.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        if raw.startswith(":"):
                            # heartbeat/comment
                            continue
                        if raw.startswith("event:"):
                            if raw[6:].strip() == "done":
                                break
                            continue
                        if raw.startswith("data: "):
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            data_str = raw[6:]
                            try:
                                payload = json.loads(data_str)
                                piece = payload.get("content", "")
                                nbytes += len(piece.encode("utf-8"))
                            except json.JSONDecodeError:
                                # potongan data tidak lengkap — abaikan
                                pass
        except Exception as e:
            exc = e
        finally:
            total = time.perf_counter() - start

            # Fire custom sample untuk TTFT
            events.request.fire(
                request_type="SSE",
                name="chat_ttft",
                response_time=(ttft * 1000 if ttft is not None else total * 1000),
                response_length=0,
                exception=exc,
                context={"status_code": status_code},
            )
            # Fire custom sample untuk total
            events.request.fire(
                request_type="SSE",
                name="chat_total",
                response_time=total * 1000,
                response_length=nbytes,
                exception=exc,
                context={"status_code": status_code},
            )