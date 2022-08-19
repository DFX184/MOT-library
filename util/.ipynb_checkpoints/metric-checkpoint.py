import motmetrics as mm
import numpy as np

def convert_motchallenge(frame_id,results):
     return np.array([",".join(
       [
            str(frame_id),
            str(int(results[i,0])), ## user id
            str(round(results[i,1],2)),
            str(round(results[i,2],2)),
            str(round(results[i,3] - results[i,1],2)),
            str(round(results[i,4] - results[i,2],2)),
            "-1",
            "-1",
            "-1",
            "-1"
        ]
    )
      for i in range(len(results))
     ],dtype = np.str_)
    

def evaluate_mot(predict_filename,truth_filename,fmt = "mot16"):
    gt = mm.io.loadtxt(truth_filename, fmt=fmt)
    ts = mm.io.loadtxt(predict_filename, fmt=fmt)
    acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou',distth=0.5)
    mh = mm.metrics.create()
    
    summary = mh.compute_many([acc],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['full'])

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    return strsummary
