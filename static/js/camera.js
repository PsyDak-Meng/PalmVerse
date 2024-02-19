
document.addEventListener('DOMContentLoaded', function () {
    var video = document.getElementById('video_feed');

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
            })
            .catch(function (error) {
                console.log("Error accessing camera: ", error);
            });
    }
});
