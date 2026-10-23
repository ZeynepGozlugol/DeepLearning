import cv2

def draw_dashboard(frame, focus_seconds, distraction_seconds, is_distracted, person_present):
    """
    Kamera karesinin üzerine odak süresi, dikkat dağınıklığı süresi
    ve durum uyarılarını şık bir panel olarak çizer.
    """
    h, w, _ = frame.shape

    # 1. Sol üst köşeye yarı saydam bilgi paneli arka planı
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (340, 160), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 2. İstatistik Metinleri
    total_time = focus_seconds + distraction_seconds
    focus_rate = (focus_seconds / total_time * 100) if total_time > 0 else 100.0

    cv2.putText(frame, "ODAK TAKIP PANELI", (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Odak Suresi: {focus_seconds:.1f} sn", (35, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Telefonda Gecen: {distraction_seconds:.1f} sn", (35, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(frame, f"Odak Basarisi: %{focus_rate:.1f}", (35, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # 3. Durum Bildirimleri ve Kırmızı Uyarı Efekti
    if not person_present:
        # Masada kimse yoksa
        cv2.putText(frame, "DURUM: Masada Kimse Yok", (w // 2 - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif is_distracted:
        # Telefonla oynanıyorsa: Kırmızı çerçeve ve büyük uyarı
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 12)
        cv2.putText(frame, "UYARI: TELEFONU BIRAK, DERSE ODAKLAN!", (w // 2 - 270, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    else:
        # Normal çalışma durumu
        cv2.putText(frame, "DURUM: Odaklanildi", (w - 230, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame