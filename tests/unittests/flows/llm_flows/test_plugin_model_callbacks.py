# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _ADK_AGENT_NAME_LABEL_KEY
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from google.genai.errors import ClientError
import pytest

from ... import testing_utils

mock_error = ClientError(
    code=429,
    response_json={
        'error': {
            'code': 429,
            'message': 'Quota exceeded.',
            'status': 'RESOURCE_EXHAUSTED',
        }
    },
)


class MockPlugin(BasePlugin):
  before_model_text = 'before_model_text from MockPlugin'
  after_model_text = 'after_model_text from MockPlugin'
  on_model_error_text = 'on_model_error_text from MockPlugin'

  def __init__(self, name='mock_plugin'):
    self.name = name
    self.enable_before_model_callback = False
    self.enable_after_model_callback = False
    self.enable_on_model_error_callback = False
    # Records the request observed by on_model_request_callback.
    self.on_model_request_count = 0
    self.observed_labels: dict[str, str] = {}
    self.observed_texts: list[str] = []
    self.observed_request_ids: list[int] = []
    self.before_model_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.before_model_text)]
        )
    )
    self.after_model_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.after_model_text)]
        )
    )
    self.on_model_error_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=self.on_model_error_text)]
        )
    )

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    if not self.enable_before_model_callback:
      return None
    return self.before_model_response

  async def on_model_request_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    # Read-only observation of the finalized request.
    self.on_model_request_count += 1
    self.observed_request_ids.append(id(llm_request))
    self.observed_labels = dict(
        (llm_request.config and llm_request.config.labels) or {}
    )
    self.observed_texts = [
        part.text
        for content in llm_request.contents or []
        for part in content.parts or []
        if part.text
    ]

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    if not self.enable_after_model_callback:
      return None
    return self.after_model_response

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    if not self.enable_on_model_error_callback:
      return None
    return self.on_model_error_response


CANONICAL_MODEL_CALLBACK_CONTENT = 'canonical_model_callback_content'


def canonical_agent_model_callback(**kwargs) -> Optional[LlmResponse]:
  return LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text=CANONICAL_MODEL_CALLBACK_CONTENT)]
      )
  )


@pytest.fixture
def mock_plugin():
  return MockPlugin()


def test_before_model_callback_with_plugin(mock_plugin):
  """Tests that the model response is overridden by before_model_callback from the plugin."""
  responses = ['model_response']
  mock_model = testing_utils.MockModel.create(responses=responses)
  mock_plugin.enable_before_model_callback = True
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', mock_plugin.before_model_text),
  ]


def test_before_model_fallback_canonical_callback(mock_plugin):
  """Tests that when plugin returns empty response, the model response is overridden by the canonical agent model callback."""
  responses = ['model_response']
  mock_plugin.enable_before_model_callback = False
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=canonical_agent_model_callback,
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', CANONICAL_MODEL_CALLBACK_CONTENT),
  ]


def test_before_model_callback_fallback_model(mock_plugin):
  """Tests that the model response is executed normally when both plugin and canonical agent model callback return empty response."""
  responses = ['model_response']
  mock_plugin.enable_before_model_callback = False
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'model_response'),
  ]


def test_on_model_error_callback_with_plugin(mock_plugin):
  """Tests that the model error is handled by the plugin."""
  mock_model = testing_utils.MockModel.create(error=mock_error, responses=[])
  mock_plugin.enable_on_model_error_callback = True
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])

  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', mock_plugin.on_model_error_text),
  ]


def test_on_model_error_callback_fallback_to_runner(mock_plugin):
  """Tests that the model error is not handled and falls back to raise from runner."""
  mock_model = testing_utils.MockModel.create(error=mock_error, responses=[])
  mock_plugin.enable_on_model_error_callback = False
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  try:
    testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  except Exception as e:
    assert e == mock_error


AGENT_MUTATION_TEXT = 'mutation_from_agent_before_model_callback'


def mutating_before_model_callback(
    *, callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
  """Agent before_model_callback that mutates the request, then proceeds."""
  llm_request.contents.append(
      types.Content(
          role='user',
          parts=[types.Part.from_text(text=AGENT_MUTATION_TEXT)],
      )
  )
  return None


def test_on_model_request_observes_finalized_request(mock_plugin):
  """on_model_request sees the agent-callback mutation and the injected label."""
  mock_model = testing_utils.MockModel.create(responses=['model_response'])
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_model_callback=mutating_before_model_callback,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'model_response'),
  ]

  # The hook fired exactly once for the single model call.
  assert mock_plugin.on_model_request_count == 1
  # It observed the mutation made by the agent's before_model_callback, which
  # runs AFTER the plugin before_model_callback.
  assert AGENT_MUTATION_TEXT in mock_plugin.observed_texts
  # It observed the agent-name label, which ADK injects AFTER all callbacks.
  assert (
      mock_plugin.observed_labels.get(_ADK_AGENT_NAME_LABEL_KEY) == 'root_agent'
  )


def test_on_model_request_skipped_when_before_model_short_circuits(mock_plugin):
  """No request is sent on short-circuit, so the observer hook must NOT fire."""
  mock_plugin.enable_before_model_callback = True
  mock_model = testing_utils.MockModel.create(responses=['model_response'])
  agent = Agent(
      name='root_agent',
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', mock_plugin.before_model_text),
  ]

  assert mock_plugin.on_model_request_count == 0


def test_on_model_request_observes_request_sent_on_live_path(mock_plugin):
  """On the live/CFC path the observer sees the same object passed to connect."""
  mock_model = testing_utils.MockModel.create(responses=['live_response'])
  agent = Agent(name='root_agent', model=mock_model)

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_content(
      types.Content(role='user', parts=[types.Part.from_text(text='hi')])
  )
  runner.run_live(live_request_queue)

  # Single connection, no reconnect: the hook fires exactly once. (It sits in
  # run_live's reconnect loop, so a resumed session would re-fire per connect().)
  assert mock_plugin.on_model_request_count == 1
  # run_live builds a fresh LlmRequest and passes it to llm.connect(); the
  # observer must see that exact object, not the outer _call_llm_async request.
  assert mock_plugin.observed_request_ids[-1] == id(mock_model.requests[-1])
  # Narrowed live/CFC contract: run_live's request is not finalized, so it
  # carries no adk_agent_name label (unlike the normal path). See issue #6222.
  assert _ADK_AGENT_NAME_LABEL_KEY not in mock_plugin.observed_labels


@pytest.mark.asyncio
async def test_on_model_request_live_cfc_is_not_finalized(mock_plugin):
  """CFC path: the observer sees run_live's fresh, un-finalized request.

  With support_cfc=True, `_call_llm_async` runs before_model_callback and injects
  the adk_agent_name label on the OUTER request, then `run_live()` builds a fresh
  LlmRequest and sends that. Per the documented contract the observer sees the
  fresh object (no adk_agent_name), which diverges from the finalized outer
  request. This pins the narrowed live/CFC contract (issue #6222); making the
  live path finalize would be a deliberate, test-visible change.
  """
  live_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='live_cfc_response')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(responses=[live_response])
  agent = Agent(name='root_agent', model=mock_model)

  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(support_cfc=True, streaming_mode=StreamingMode.SSE),
      plugins=[mock_plugin],
  )
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author='root_agent',
  )

  outer_request = LlmRequest(model='mock')
  flow = BaseLlmFlow()
  responses = []
  async with Aclosing(
      flow._call_llm_async(
          invocation_context, outer_request, model_response_event
      )
  ) as agen:
    async for resp in agen:
      responses.append(resp)
      # The first live response is enough; the observer has already fired before
      # connect(). Breaking avoids waiting on run_live's empty send queue;
      # Aclosing() finalizes run_live (cancelling its send task).
      break

  assert len(responses) >= 1
  # The hook fired once for the single live connection.
  assert mock_plugin.on_model_request_count == 1
  # It observed the exact fresh object passed to llm.connect().
  assert mock_plugin.observed_request_ids[-1] == id(mock_model.requests[-1])
  # The OUTER request WAS finalized with the label...
  assert (
      outer_request.config.labels.get(_ADK_AGENT_NAME_LABEL_KEY) == 'root_agent'
  )
  # ...but the observed (live) request was NOT — the divergence #6222 flags.
  assert _ADK_AGENT_NAME_LABEL_KEY not in mock_plugin.observed_labels


@pytest.mark.asyncio
async def test_on_model_request_only_fires_for_sent_calls_under_call_limit(
    mock_plugin,
):
  """With max_llm_calls=1, the blocked 2nd call must not reach the observer."""

  def repeat_tool() -> dict:
    return {'status': 'ok'}

  # 1st response is a function call (triggers a 2nd model call); the 2nd model
  # call is rejected by the call-count guard before it is sent.
  responses = [
      types.Part(function_call=types.FunctionCall(name='repeat_tool', args={})),
      'final_response',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(name='root_agent', model=mock_model, tools=[repeat_tool])

  runner = testing_utils.InMemoryRunner(agent, plugins=[mock_plugin])
  session = runner.session
  try:
    async for _ in runner.runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=testing_utils.get_user_content('test'),
        run_config=RunConfig(max_llm_calls=1),
    ):
      pass
  except LlmCallsLimitExceededError:
    pass

  # Only the first, actually-sent request reached the observer. (Before the fix
  # the observer fired before the guard and would have counted the blocked call.)
  assert mock_plugin.on_model_request_count == 1


def test_on_model_request_exception_does_not_abort_call():
  """A raising observer is logged and swallowed; the model call still runs."""

  class RaisingPlugin(BasePlugin):

    def __init__(self):
      self.name = 'raising_plugin'

    async def on_model_request_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
      raise ValueError('boom')

  mock_model = testing_utils.MockModel.create(responses=['model_response'])
  agent = Agent(name='root_agent', model=mock_model)

  runner = testing_utils.InMemoryRunner(agent, plugins=[RaisingPlugin()])
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', 'model_response'),
  ]


if __name__ == '__main__':
  pytest.main([__file__])
