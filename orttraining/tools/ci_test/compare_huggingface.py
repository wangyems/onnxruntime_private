import sys
import json
import pytest

actual = sys.argv[1]
expect = sys.argv[2]

with open(actual) as file_actual:
  json_actual = json.loads(file_actual.read())

with open(expect) as file_expect:
  json_expect = json.loads(file_expect.read())

# loss curve match
def almost_equal(x, y, threshold=0.0001):
  print(abs(x-y))
  return abs(x-y) < threshold

for i in range(len(json_actual['steps'])): 
  step_actual = json_actual['steps'][i]
  step_expect = json_expect['steps'][i]
  print(step_actual['step'])
  assert(step_actual['step'] == step_expect['step'])
  assert(almost_equal(step_actual['loss'], step_expect['loss']))

# perf match
assert(json_actual['samples_per_second'] >= 0.95*json_expect['samples_per_second'])
