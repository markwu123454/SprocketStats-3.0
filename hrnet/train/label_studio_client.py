"""LabelStudio client with saner pagination defaults for this project's scale (~112k tasks).

The stock SDK's tasks.list() defaults to fields="all" (full annotation/prediction
payload on every task) with no only_annotated filter and no page_size, so a bare
`client.tasks.list(project=...)` walks the entire project page-by-page fetching
everything -- which hangs/times out here. This subclasses the SDK's own extension
point (TasksClientExt) to set better defaults; callers can still override them.
"""
from label_studio_sdk import LabelStudio
from label_studio_sdk.tasks.client_ext import TasksClientExt


class TasksClientDefaults(TasksClientExt):
    def list(self, **kwargs):
        kwargs.setdefault("only_annotated", True)
        kwargs.setdefault("page_size", 200)
        return super().list(**kwargs)

    list.__doc__ = TasksClientExt.list.__doc__


class LabelStudioClient(LabelStudio):
    @property
    def tasks(self) -> TasksClientDefaults:
        if self._tasks_ext is None:
            self._tasks_ext = TasksClientDefaults(client_wrapper=self._client_wrapper)
        return self._tasks_ext
