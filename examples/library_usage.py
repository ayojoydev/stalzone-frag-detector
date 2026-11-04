"""Use the library without CLI parsing, a console menu or a graphical interface."""

from szfd import DetectorOptions, analyze, export


def make_moments(video_path: str, output_directory: str):
    options = DetectorOptions(
        multikill=True,
        multifrag_gap=10,
        plaque_duration=5,
        cut_before=5,
        cut_after=3,
        cut_mode="exact",
    )
    analysis = analyze(video_path, options)
    for moment in analysis.clip_groups:
        print(f"{moment.start_seconds:.2f}–{moment.end_seconds:.2f} sec")
    # Analyze does not create files. Export only when desired.
    return export(analysis, output_directory)
