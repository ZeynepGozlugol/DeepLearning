import os
import threading
import cv2
from ultralytics import YOLO

# Mac (macOS) yerleşik sistem uyarı sesi
def play_beep():
    # macOS'un hazır gelen sistem uyarı sesini çalar
    os.system('afplay /System/Library/Sounds/Tink.aiff')

def run_immediate_phone_alert():
    # En hızlı nano model
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Kamera açılamadı. macOS Ayarlar > Gizlilik ve Güvenlik > Kamera altından PyCharm'a izin verildiğinden emin olun.")
        return

    # COCO sınıfı: 67 -> cell phone (cep telefonu)
    PHONE_CLASS_ID = 67

    # Sesin üst üste binmesini önleyen bayrak
    is_beep_playing = False

    def trigger_sound():
        nonlocal is_beep_playing
        is_beep_playing = True
        play_beep()
        is_beep_playing = False

    print("Sistem devrede... Çıkmak için kamera penceresindeyken 'q' tuşuna basın.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sadece cep telefonu için arama yap
        results = model(frame, classes=[PHONE_CLASS_ID], verbose=False)

        phone_detected = False

        # Telefon tespit edildi mi kontrol et
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.45:
                phone_detected = True
                break

        # Modelin çizdiği kutuları al
        annotated_frame = results[0].plot()
        h, w, _ = annotated_frame.shape

        # TELEFON GÖRÜNDÜĞÜ AN ÇALIŞACAK MANTIK
        if phone_detected:
            # 1. Sesli Uyarı: Thread ile arka planda anında çal
            if not is_beep_playing:
                threading.Thread(target=trigger_sound, daemon=True).start()

            # 2. Ekranın etrafına kalın kırmızı alarm çerçevesi
            cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 0, 255), 18)

            # 3. Ortaya büyük kırmızı uyarı bandı
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (20, h // 2 - 50), (w - 20, h // 2 + 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

            cv2.putText(
                annotated_frame,
                "UYARI: TELEFONU DERHAL BIRAK!",
                (w // 2 - 280, h // 2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                3
            )
        else:
            # Telefon yokken yeşil durum bildirimi
            cv2.putText(
                annotated_frame,
                "DURUM: ODAKLANILDI (TEMIZ)",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2
            )

        cv2.imshow("Odak Kontrol - Telefon Yasak", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_immediate_phone_alert()