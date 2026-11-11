"""Recover interrupted asset replacement without guessing or discarding evidence."""
import json
import os
from pathlib import Path
import shutil
import uuid
import time
from contextlib import contextmanager

from src.core.replay_journal import atomic_json
from src.core.replay_evidence import content_sha256
from src.hardware.render_cache import _manifest_path,file_identity


class DeliveryRecoveryError(RuntimeError):
    pass


def _identity(path):
    path=Path(path)
    if path.is_symlink():raise DeliveryRecoveryError(f'Unsafe delivery asset: {path}')
    if not path.exists():return None
    if path.is_symlink() or not path.is_file():raise DeliveryRecoveryError(f'Unsafe delivery asset: {path}')
    before=path.stat()
    result={'size':before.st_size,'sha256':content_sha256(path)}
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise DeliveryRecoveryError(f'Delivery asset changed during inspection: {path}')
    return result


@contextmanager
def _io_admission():
    from src.hardware.scheduler import get_disk_semaphore,acquire_with_retry
    disk=get_disk_semaphore()
    if not acquire_with_retry(disk,timeout=10,retries=2):
        raise TimeoutError('Delivery recovery I/O admission timed out')
    try:yield
    finally:disk.release()


def prepare_transaction(backup,output,sources,targets,*,fingerprint=None,context=None,keep_previous=False):
    with _io_admission():
        return _prepare_transaction(backup,output,sources,targets,fingerprint=fingerprint,context=context,keep_previous=keep_previous)


def _prepare_transaction(backup,output,sources,targets,*,fingerprint=None,context=None,keep_previous=False):
    if any(not target.resolve().is_relative_to(output.parent.resolve()) for target in targets):
        raise DeliveryRecoveryError('Delivery target escapes output directory')
    journal={'version':1,'state':'PREPARED','output':str(output.resolve()),'context':context,
             'fingerprint':fingerprint,'keep_previous':keep_previous,'assets':[]}
    candidates={str(target.resolve()):source for source,target in sources}
    for index,target in enumerate(targets):
        source=candidates.get(str(target.resolve()))
        if source:
            with source.open('r+b') as stream:os.fsync(stream.fileno())
        journal['assets'].append({'target':str(target.resolve()),'backup':str(index),
                                  'old':_identity(target),'new':_identity(source) if source else None})
    atomic_json(backup/'transaction.json',journal)
    return journal


def commit_transaction(backup,journal):
    journal={**journal,'state':'COMMITTED','committed_at':time.time()}
    atomic_json(backup/'transaction.json',journal)


def pending_transactions(directory, *, context=None, output=None):
    pending=[]
    for backup in Path(directory).glob('.delivery_backup_*'):
        if backup.is_symlink():raise DeliveryRecoveryError(f'Linked recovery directory requires explicit inspection: {backup}')
        path=backup/'transaction.json'
        if not path.exists():continue  # Legacy packages require explicit manual inspection.
        try:journal=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError) as error:
            raise DeliveryRecoveryError(f'Unreadable delivery journal: {path}') from error
        if journal.get('state') in ('COMMITTED','RECOVERED'):continue
        if context is not None and journal.get('context')!=context:continue
        if output is not None and os.path.normcase(journal.get('output',''))!=os.path.normcase(str(Path(output).resolve())):continue
        pending.append(backup)
    return pending


def reject_legacy_recovery(directory, output=None):
    """Read-only guard, including before a historical COMPLETED shortcut.

    A missing recorded output cannot safely disambiguate legacy packages.
    Successful review history and versioned journals use their existing paths.
    """
    for backup in Path(directory).glob('.delivery_backup_*'):
        if backup.is_symlink():
            raise DeliveryRecoveryError(f'Linked recovery directory requires explicit inspection: {backup}')
        if (backup/'transaction.json').exists() or (backup/'review_history.json').exists():
            continue
        legacy = backup/'recovery.json'
        if not legacy.exists():
            continue
        try:
            mapping = json.loads(legacy.read_text(encoding='utf-8'))
            destination = mapping['0']
            if not isinstance(destination, str) or not destination:
                raise ValueError('Missing legacy destination')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DeliveryRecoveryError(f'Unreadable legacy recovery map: {legacy}') from error
        if output is None or os.path.normcase(destination) == os.path.normcase(str(Path(output).resolve())):
            raise DeliveryRecoveryError(f'Legacy interrupted package needs explicit inspection: {backup}')


def _copy_durable(source,target,expected=None):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.recovery.tmp')
    try:
        with Path(source).open('rb') as reader,temporary.open('wb') as writer:
            shutil.copyfileobj(reader,writer,1024*1024)
            writer.flush();os.fsync(writer.fileno())
        shutil.copystat(source,temporary)
        if expected is not None and _identity(temporary)!=expected:
            raise DeliveryRecoveryError(f'Recovery copy identity mismatch: {source}')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)


def recover_transaction(backup, *, mark_pending=None):
    with _io_admission():
        return _recover_transaction(backup,mark_pending=mark_pending)


def _recover_transaction(backup, *, mark_pending=None):
    """Caller owns the day/workspace/output lease. Validate all assets before writes."""
    backup=Path(backup)
    if not backup.name.startswith('.delivery_backup_') or backup.is_symlink():
        raise DeliveryRecoveryError('Invalid recovery directory')
    backup=backup.resolve()
    path=backup/'transaction.json';journal=json.loads(path.read_text(encoding='utf-8'))
    output=Path(journal['output'])
    targets=[output,output.with_suffix('.meta.json'),output.with_suffix('.srt'),_manifest_path(output)]
    if (journal.get('version')!=1 or output.parent.resolve()!=backup.parent
            or len(journal.get('assets',[]))!=4):
        raise DeliveryRecoveryError('Recovery journal binding mismatch')
    if journal['state'] in ('COMMITTED','RECOVERED'):return journal['state']
    if journal['state']!='PREPARED':raise DeliveryRecoveryError('Unrecognized transaction state')
    if any(not target.resolve().is_relative_to(backup.parent) for target in targets):
        raise DeliveryRecoveryError('Recovery target escapes output directory')
    checks=[]
    for index,entry in enumerate(journal['assets']):
        target=targets[index]
        if entry['target']!=str(target.resolve()) or entry['backup']!=str(index):
            raise DeliveryRecoveryError('Recovery asset binding mismatch')
        old,new=entry['old'],entry['new'];saved=backup/str(index)
        current=_identity(target);saved_identity=_identity(saved)
        if saved_identity is not None and saved_identity!=old:
            raise DeliveryRecoveryError(f'Previous asset changed: {saved}')
        if old is not None and saved_identity is None and current!=old:
            raise DeliveryRecoveryError(f'Previous asset unavailable: {target}')
        is_new=current is not None and current!=old and current==new
        if index==3 and not is_new and current is not None and current!=old and journal.get('fingerprint'):
            manifest=json.loads(target.read_text(encoding='utf-8'))
            video_identity=journal['assets'][0]['new']
            is_new=(manifest.get('fingerprint')==journal['fingerprint'] and video_identity is not None
                    and manifest.get('output')==file_identity(output)
                    and _identity(output)==video_identity)
            if is_new:
                entry['new']=current  # Persist manifest identity before any recovery write.
        if current is not None and current!=old and not is_new:
            raise DeliveryRecoveryError(f'Unrecognized current asset; retained for inspection: {target}')
        interrupted=backup/'interrupted'/str(index)
        if interrupted.exists() and is_new and _identity(interrupted)!=current:
            raise DeliveryRecoveryError(f'Conflicting interrupted evidence: {interrupted}')
        checks.append((target,saved,interrupted,old,current,is_new))
    atomic_json(path,journal)
    for target,saved,interrupted,old,current,is_new in checks:
        if is_new and not interrupted.exists():_copy_durable(target,interrupted,expected=current)
    for target,saved,interrupted,old,current,is_new in checks:
        if _identity(target)!=current:
            raise DeliveryRecoveryError(f'Asset changed before recovery: {target}')
        if old is not None and saved.exists():
            if current!=old or target.stat().st_mtime_ns!=saved.stat().st_mtime_ns:
                _copy_durable(saved,target,expected=old)
        elif old is None and is_new:target.unlink()
    if mark_pending is not None:mark_pending()
    atomic_json(path,{**journal,'state':'RECOVERED'})
    return 'RECOVERED'


def recover_output(output, *, mark_pending=None):
    reject_legacy_recovery(Path(output).parent, output)
    repaired=0
    for backup in Path(output).parent.glob('.delivery_backup_*'):
        if backup.is_symlink():raise DeliveryRecoveryError(f'Linked recovery directory requires explicit inspection: {backup}')
        transaction=backup/'transaction.json'
        if transaction.exists():
            journal=json.loads(transaction.read_text(encoding='utf-8'))
            if (journal.get('state')=='COMMITTED' and journal.get('keep_previous')
                    and os.path.normcase(journal.get('output',''))==os.path.normcase(str(Path(output).resolve()))
                    and not (backup/'review_history.json').exists()):
                with _io_admission():
                    assets={}
                    if journal.get('version')!=1 or len(journal.get('assets',[]))!=4:
                        raise DeliveryRecoveryError('Invalid committed review journal')
                    for index,entry in enumerate(journal['assets']):
                        if entry['backup']!=str(index):raise DeliveryRecoveryError('Invalid history binding')
                        if entry['old'] is not None:
                            if _identity(backup/str(index))!=entry['old']:
                                raise DeliveryRecoveryError('Previous review asset changed')
                            assets[Path(entry['target']).name]=str(index)
                    atomic_json(backup/'review_history.json',{'version':1,'output':journal['output'],
                                'committed_at':journal['committed_at'],'assets':assets})
                    repaired+=1
            continue
        if (backup/'review_history.json').exists():continue
    pending=pending_transactions(Path(output).parent,output=output)
    if len(pending)>1:raise DeliveryRecoveryError('Multiple interrupted packages; retain for explicit inspection')
    for backup in pending:recover_transaction(backup,mark_pending=mark_pending)
    return len(pending)+repaired
