"""
A simple IoU-based tracker: matches detections across frames by box overlap
(greedy matching on IoU), assigns new IDs to unmatched detections, and drops
tracks that go too many frames without a matching detection.

This is intentionally simple (no motion prediction, no Hungarian algorithm)
so it's easy to understand and explain in a report -- appropriate for the
scope of this assignment.
"""


class Track:
    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box          # [xmin, ymin, xmax, ymax]
        self.missed_frames = 0
        self.counted = False    # whether this track has already been counted


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_missed_frames=10):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """
        detections: list of [xmin, ymin, xmax, ymax]
        Returns the current list of active Track objects (after matching).
        """
        matched_track_idxs = set()
        matched_det_idxs = set()

        # greedy matching: for each existing track, find the best unmatched detection
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = iou(track.box, det)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, ti, di in pairs:
            if ti in matched_track_idxs or di in matched_det_idxs:
                continue
            self.tracks[ti].box = detections[di]
            self.tracks[ti].missed_frames = 0
            matched_track_idxs.add(ti)
            matched_det_idxs.add(di)

        # unmatched existing tracks: increment missed-frame counter
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idxs:
                track.missed_frames += 1

        # unmatched detections: start new tracks
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                new_track = Track(self._next_id, det)
                self._next_id += 1
                self.tracks.append(new_track)

        # drop tracks that have been missing too long
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed_frames]

        return self.tracks
