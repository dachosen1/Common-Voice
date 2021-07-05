    /* Audio recording and streaming demo by Miguel Grinberg.
   Adapted from https://webaudiodemos.appspot.com/AudioRecorder
   Copyright 2013 Chris Wilson
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;
var socketio = io.connect(location.origin + '/', {transports: ['websocket']});
socketio.on('gender_model', function (ms) {
        $('div.gender-box').text(ms.pred).html()
        $('div.category-prob-gender').text(ms.prob).html()
      })
    socketio.on('connect', function () {
        //socket.emit('audio-streaming', {})
        console.log('Connected', socketio.id)
      })

document.getElementById("mic").addEventListener("click", function(e) {
   
    toggleRecording(e);
});

function toggleRecording( e ) {
    if (e.target.classList.contains('recording')) {
        // stop recording
        e.target.classList.remove('recording');
        recording = false;
        socketio.emit('end-recording');
    } else {
        // start recording
        e.target.classList.add('recording');
        recording = true;
        socketio.emit('start-audio', {numChannels: 1, bps: 16, fps: parseInt(audioContext.sampleRate)});
    }
}

function convertToMono( input ) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect( splitter );
    splitter.connect( merger, 0, 0 );
    splitter.connect( merger, 0, 1 );
    return merger;
}



function gotStream(stream) {
     document.getElementById("prediction-result").classList.add("prediction-show");
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;

    audioInput = convertToMono( audioInput );
    audioInput.connect(inputPoint);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect( analyserNode );

    scriptNode = (audioContext.createScriptProcessor || audioContext.createJavaScriptNode).call(audioContext, 1024, 1, 1);
    scriptNode.onaudioprocess = function (audioEvent) {

            input = audioEvent.inputBuffer.getChannelData(0);

            // convert float audio data to 16-bit PCM
            var buffer = new ArrayBuffer(input.length * 2)
            var output = new DataView(buffer);
            for (var i = 0, offset = 0; i < input.length; i++, offset += 2) {
                var s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            socketio.emit('stream-audio', buffer);

    }
    inputPoint.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect( zeroGain );
    zeroGain.connect( audioContext.destination );
    //updateAnalysers();
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    
    navigator.getUserMedia({audio: true}, gotStream, function(e) {
        alert('Error getting audio');
        document.getElementById("prediction-result").classList.remove("prediction-show");
        console.log(e);
    });
    
}

window.addEventListener('load', initAudio );