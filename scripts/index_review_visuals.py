"""Build/rebuild derived visual features on a chosen database without changing labels.

Historical backfill samples midpoint frames, with actual PTS and source identity.
Future full analyses reuse their existing decoded gray frames instead.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",required=True,type=Path)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--date")
    parser.add_argument("--source-file", help="只处理库中登记的这条原片路径")
    parser.add_argument("--source-budget-s", type=float, default=60, help="单源读取截止预算，范围1–1800秒；超时终止独立读者")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true", help="重试已记录的缺帧/失败源；默认先推进尚未尝试的源")
    parser.add_argument("--cam-index",type=int)
    parser.add_argument("--limit-files",type=int,default=20)
    args = parser.parse_args()
    if (args.limit_files < 1 or args.cam_index is not None and args.cam_index < 0
            or not math.isfinite(args.source_budget_s) or not 1 <= args.source_budget_s <= 1800):
        parser.error("Invalid scope")
    if not args.db.is_file():
        parser.error("Database must already exist; refusing to create an empty inventory")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.database import VlogDatabase
    from src.core.replay_journal import atomic_json, run_lease
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with run_lease(args.report.parent):
        report = json.loads(args.report.read_text(encoding='utf-8')) if args.report.exists() else {"scope":"derived visual features only; not independent event truth","runs":[]}
        if any(entry.get('db') != str(args.db.resolve()) for entry in report['runs']):
            parser.error("Report belongs to a different database")
        return run_inventory(args, report, VlogDatabase, atomic_json)


def run_inventory(args, report, database_factory, atomic_json):
    from src.core.visual_index import index_source, VisualReaderStopError,confirm_reader_stopped
    resolutions=[]
    for earlier in report['runs']:
        if earlier.get('active_reader'):
            resolutions.append({'reader':earlier['active_reader'],'observed':confirm_reader_stopped(earlier['active_reader'])})
        elif earlier.get('status')=='READER_STOP_FAILED':
            raise VisualReaderStopError('Legacy reader-stop failure lacks process identity; inspect and record recovery evidence before continuing')
    db = database_factory(db_path=args.db)
    conditions,parameters = [],[]
    for field,value in (("date",args.date),("cam_index",args.cam_index),("filepath",args.source_file)):
        if value is not None:
            conditions.append(field+"=?");parameters.append(value)
    where = " WHERE "+" AND ".join(conditions) if conditions else ""
    files = [row[0] for row in db.conn.execute("SELECT filepath FROM segments"+where+" GROUP BY filepath ORDER BY MAX(needs_review) DESC,MAX(date) DESC,filepath",parameters)]
    previous = {entry['filepath']: entry for earlier in report['runs'] for entry in earlier['files']}
    # New sources first: a permanently unreadable source must not consume
    # every subsequent bounded run before the remaining inventory is reached.
    files.sort(key=lambda path: path in previous)
    if args.source_file and not files:
        db.close()
        raise ValueError('Source file is not registered in the selected inventory scope')
    run = {"started_at":time.time(),"db":str(args.db.resolve()),"scope":{"date":args.date,"cam_index":args.cam_index,"source_file":args.source_file},
           "source_budget_s":args.source_budget_s,"files":[],"source_inventory":len(files),"status":"RUNNING",'reader_resolutions':resolutions}
    report["runs"].append(run)
    def persist():
        atomic_json(args.report, report)
    try:
        persist()
        for filepath in files:
            if len(run["files"]) >= args.limit_files:
                break
            earlier = previous.get(filepath)
            if (earlier and earlier['status']=='FAILED' and not earlier.get('source_identity')
                    and not args.force and not args.retry_incomplete):
                continue
            identity = layout = None
            run['active_source']=filepath;persist()
            def reader_started(identity):
                run['active_reader']=identity;persist()
            try:
                result = {"filepath":filepath,**index_source(db,filepath,reuse_current=not args.force,
                        timeout_s=args.source_budget_s,previous=earlier,retry_incomplete=args.retry_incomplete,on_reader_start=reader_started)}
                run.pop('active_reader',None)
                if result.get('deferred'):
                    run.pop('active_source',None);persist()
                    continue
                result['status'] = 'COMPLETED' if result['indexed'] == result['segments'] else 'PARTIAL'
                # A matching ledger never substitutes for the actual current
                # derived rows: index_source verifies them or rebuilds them.
                if earlier and earlier['status'] == 'COMPLETED' and result.get('cache_hit') and not args.force:
                    run.pop('active_source',None);persist()
                    continue
            except (InterruptedError, VisualReaderStopError):
                raise  # Cancellation is not an isolated source failure to continue past.
            except Exception as exc:
                run.pop('active_reader',None)
                identity=getattr(exc,'source_identity',None);layout=getattr(exc,'segment_layout_sha256',None)
                result = {"filepath":filepath,"status":"FAILED","error":str(exc),
                          "source_identity":identity,"segment_layout_sha256":layout}
            run["files"].append(result);run.pop('active_source',None);persist()
            print(json.dumps({"source":Path(filepath).name,"status":result["status"],"indexed":result.get("indexed")},ensure_ascii=True),flush=True)
        latest = {entry['filepath']: entry for earlier in report['runs'] for entry in earlier['files']}
        run.update(finished_at=time.time(), status='INCOMPLETE',
                   recorded_complete_sources=sum(latest.get(path,{}).get('status')=='COMPLETED' for path in files),
                   recorded_partial_sources=sum(latest.get(path,{}).get('status')=='PARTIAL' for path in files),
                   recorded_failed_sources=sum(latest.get(path,{}).get('status')=='FAILED' for path in files))
        if run['recorded_complete_sources'] == len(files):
            run['status']='RECORDED_COMPLETE_PENDING_LIVE_INVENTORY_ACCEPTANCE'
        persist()
        print(json.dumps({key:run[key] for key in ('status','source_inventory','recorded_complete_sources','recorded_partial_sources','recorded_failed_sources')}),flush=True)
        if any(entry['status']=='FAILED' for entry in run['files']):
            raise SystemExit(1)
    except (InterruptedError, KeyboardInterrupt) as error:
        run.update(status='INTERRUPTED',finished_at=time.time(),error=str(error));persist()
        raise
    except VisualReaderStopError as error:
        run.update(status='READER_STOP_FAILED',finished_at=time.time(),error=str(error));persist()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
