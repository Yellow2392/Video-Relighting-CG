from processing.segmentation import GreenScreenProcessor

if __name__ == "__main__":
    processor = GreenScreenProcessor(background_path='data/background/sunset.jpg')

    # Webcam:
    #processor = GreenScreenProcessor(background_path='data/background/sunset.jpg')
    #processor.process_webcam(device_id=0)

    # .mp4:
    processor.process_video_file('data/subject/object_2.mp4')

    # para guardar el resultado:
    #processor.process_video_file(video_path='data/subject/object_1.mp4', output_path='videos/resultado.mp4')