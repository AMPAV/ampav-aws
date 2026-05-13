"""AWS tooling for AMPAV."""

"""BDW: Overall Notes

You should never need to import anything from __future__ unless you are
using an experimental feature or trying to write python2 code in a python3
sort of way.

Common style is there are always two blank lines after a function or method 
definition. (I'm going to break that rule with my inline examples for brevity)

Likewise, while python code tends to lean toward top-down design, there comes
a point where it becomes more distracting than helpful.  One rule of thumb is
if the documentation string is longer than your implementation, then you should
probably just leave it inline with a comment, especially if you're only using
it once.  There are exceptions, but they're usually related to if you're doing
something very clever.  I have a method 

    def find_by_type(self, output_type: object) -> list[str]:    
        "Get the output keys for any object that is the same as the output type"
        return [k for k, v in self.outputs.items() if type(output_type) == type(v)]

which is really just borderline:  it's a list comprehension and it's not
terribly clever...but it is something one would potentially do a bunch, so I
left it in there instead of letting other people redo it over and over.

Optional arguments are your friend -- use them instead of building argument
passing classes, and let them generate defaults if they're not provided.


One thing I'm seeing here:  There are a ton of assumptions on how this is going
to be used:  naming conventions, job name creation, output directory names,
requirement for a config file, etc.  

All of that should really be simplified so someone with a file on S3 can
run something like this:

from ampav.aws.transcribe import AWSTranscribe
import time

aws = AWSTranscribe('my-transcription-job')  # which may include boto3 client parameters too 
aws.submit('s3://audio/test.mp3', 'my-transcription-bucket', 
           'test-transcription.json', language='en-US')
aws.wait_for_completion()
# Or:
# while not aws.is_finished():
#    time.sleep(5)
t = aws.get_transcription(delete_from_s3=True)  # should also delete the job
print(t.output.text)

and get the transcription text.  No config files, no saving local files, etc. 
Just some parameters and method calls.  There should also be some flexibility
in terms of how one waits, if it raises an exception on .is_finished(), etc.

While all of those things may be used in a larger system down the road (by us
or anyone else), at this point we're just writing low level wrappers and
convenience functions/classes.

"""