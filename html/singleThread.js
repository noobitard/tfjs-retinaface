// Helper functions
function product() {
    var args = Array.prototype.slice.call(arguments); // makes array from arguments
    return args.reduce(function tl (accumulator, value) {
      var tmp = [];
      accumulator.forEach(function (a0) {
        value.forEach(function (a1) {
          tmp.push(a0.concat(a1));
        });
      });
      return tmp;
    }, [[]]);
  }

let range = n => [...Array(n).keys()]


// Model configs
cfgMobileNet = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": false,
    "loc_weight": 2.0,
    "gpu_train": true,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": true,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}

class PriorBox {
    constructor(cfg, imageSize){
        this.minSizes = cfg["min_sizes"]
        this.steps = cfg["steps"]
        this.clip = cfg["clip"]
        this.imageSize = imageSize
        this.featuresMaps = []
        for (let step of this.steps){
            let ele = [Math.ceil(this.imageSize[0] / step), Math.ceil(this.imageSize[1] / step)]
            this.featuresMaps.push(ele)
        }
        this.name = "s"
    }

    forward(){
        let anchors = []
        let k = 0
        for (let f of this.featuresMaps){
            let minSizes = this.minSizes[k]
            for (let ij of product(range(f[0]), range(f[1]))){
                let i = ij[0]
                let j = ij[1]
                for (let minSize of minSizes){
                    let sKx = minSize / this.imageSize[1]
                    let sKy = minSize / this.imageSize[0]
                    let denseCx = (j + 0.5) * this.steps[k] / this.imageSize[1]
                    let denseCy = (i + 0.5) * this.steps[k] / this.imageSize[0]

                    anchors.push([denseCx, denseCy, sKx, sKy])
                }
            }
            k += 1
        }
        let output = tf.tensor(anchors)
        output.reshape([-1, 4])
        if (this.clip) {
            output.clipByValue(0, 1)
        }
        return output
    }
}

async function decode(loc, priors, variances){
    let boxes = tf.concat([
        priors.slice([0, 0], [-1, 2]).add(loc.slice([0, 0], [-1, 2]).mul(variances[0]).mul(priors.slice([0, 2], [-1, -1]))),
        priors.slice([0, 2], [-1, -1]).mul(tf.exp(loc.slice([0, 2], [-1, -1]).mul(variances[1])))
    ], 1)

    let returnBoxes = tf.concat([
        boxes.slice([0, 0], [-1, 2]).sub(boxes.slice([0, 2], [-1, -1]).div(2)),
        boxes.slice([0, 2], [-1, -1]).add(boxes.slice([0, 0], [-1, 2]).sub(boxes.slice([0, 2], [-1, -1]).div(2)))
    ], 1)
    return returnBoxes
}

async function decodeLandm(pre, priors, variances){
    let priors1st = priors.slice([0, 0], [-1, 2])
    let priors2nd = priors.slice([0, 2], [-1, -1])
    let landms = tf.concat([
        priors1st.add(pre.slice([0, 0], [-1, 2]).mul(variances[0]).mul(priors2nd)),
        priors1st.add(pre.slice([0, 2], [-1, 2]).mul(variances[0]).mul(priors2nd)),
        priors1st.add(pre.slice([0, 4], [-1, 2]).mul(variances[0]).mul(priors2nd)),
        priors1st.add(pre.slice([0, 6], [-1, 2]).mul(variances[0]).mul(priors2nd)),
        priors1st.add(pre.slice([0, 8], [-1, 2]).mul(variances[0]).mul(priors2nd)),
    ], 1)

    return landms
}

async function nms(dets, thresh, keep){
    let boxes = dets.slice([0, 0], [-1, 4])
    let scores = dets.slice([0, 4], [-1, -1]).squeeze()
    let indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, keep, thresh)
    return indices.arraySync()
}

async function parseDet(det){
    landmarks = det.slice(5, -1).reshape([5, 2])
    box = det.slice(0, 4)
    score = det.slice(4, 1).squeeze()
    return [box, landmarks, score]
}


async function postProcess(loc, conf, landms, priorData, cfg, scale, scale1, resize, confidenceThreshold, topK, nmsThreshold, keepTopK){
    let boxes = await decode(loc, priorData, cfg["variance"])
    boxes = boxes.mul(scale).div(resize)
    scores = conf.slice([0, 1], [-1, 1]).squeeze()
    let landmsCopy = await decodeLandm(landms, priorData, cfg["variance"])
    landmsCopy = landmsCopy.mul(scale1).div(resize)
    let inds = await tf.whereAsync(scores.greater(confidenceThreshold).asType('bool'))
    inds = inds.squeeze()
    boxes = tf.gather(boxes, inds)
    landmsCopy = tf.gather(landmsCopy, inds)
    scores = tf.gather(scores, inds)

    topK = Math.min(topK, inds.shape[0])

    let order = tf.topk(scores, topK, true)["indices"]
    boxes = tf.gather(boxes, order)
    landmsCopy = tf.gather(landmsCopy, order)
    scores = tf.gather(scores, order)

    keepTopK = Math.min(keepTopK, order.shape[0])

    let dets = tf.concat([boxes, scores.expandDims(1)], 1)
    keep = await nms(dets, nmsThreshold, keepTopK)

    dets = tf.gather(dets, keep)
    landmsCopy = tf.gather(landmsCopy, keep)

    dets = tf.concat([dets, landmsCopy], 1)
    detsLen = dets.shape[0]

    results = []
    for (let i of range(detsLen)){
        let det = dets.slice(i, 1).squeeze()
        results.push(await parseDet(det))
    }
    return results
}

async function batchDetect(net, images){
    let confidenceThreshold = 0.5
    let topK = 5000
    let nmsThreshold = 0.4
    let keepTopK = 750
    let resize = 1
    let mean = tf.tensor([104.0, 117.0, 123.0]).expandDims(1).expandDims(1).expandDims(0)
    images = images.sub(mean)

    let batchSize = images.shape[0]
    let imHeight = images.shape[2]
    let imWidth = images.shape[3]
    let scale = tf.tensor([imWidth, imHeight, imWidth, imHeight])

    let results = await net.predict(images);
    let loc = results[0]
    let conf = tf.softmax(results[1], 2)
    let landms = results[2]

    priorbox = new PriorBox(cfgMobileNet, [imHeight, imWidth])
    let priors = priorbox.forward()
    let scale1 = tf.tensor([imWidth, imHeight, imWidth, imHeight, imWidth, imHeight, imWidth, imHeight, imWidth, imHeight])
    
    batchSize = loc.shape[0]
    let allDets = []
    for (let i of range(batchSize)){
        let loci = loc.slice(i, 1).squeeze()
        let confi = conf.slice(i, 1).squeeze()
        let landmsi = landms.slice(i, 1).squeeze()
        allDets.push(
            await postProcess(loci, confi, landmsi, priors, cfgMobileNet, scale, scale1, resize, confidenceThreshold, topK, nmsThreshold, keepTopK)
        )
    }
    return allDets
}


let net
(async function initNet(){
    net = await tf.loadLayersModel('/exported/model.json')
    console.log('Successfully loaded model')
})()

async function predict(net, imgEl, imgHeight, imgWidth){
    let inputTensor = tf.browser.fromPixels(imgEl)
    inputTensor = inputTensor.resizeBilinear([imgHeight, imgWidth])
    inputTensor = tf.expandDims(tf.transpose(inputTensor, [2, 0, 1]), 0)
    const result = await batchDetect(net, inputTensor)
    return result
}

function onPredict(){
    let resizeHeight = Number(document.getElementById("field1").value)
    let resizeWidth = Number(document.getElementById("field2").value)

    let imageInput = document.getElementById("field3")
    let canvas = document.getElementById("canvas")
    let ctx = canvas.getContext("2d")
    let waitingTxt = document.getElementById("waiting")
    image = new Image()
    image.onload = function(){
        if (resizeHeight == 0 || resizeWidth == 0) {
            resizeHeight = Math.floor(image.height/32)*32
            resizeWidth = Math.floor(image.width/32)*32
        }

        if (resizeHeight % 32 != 0 || resizeWidth % 32 != 0){
            alert("The size must be divisible by 32!")
            return
        }

        canvas.width = resizeWidth
        canvas.height = resizeHeight
        ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, resizeWidth, resizeHeight)
        predict(net, image, resizeHeight, resizeWidth).then(
            results => {
                waitingTxt.style.visibility = "hidden"
                for (let result of results[0]){
                    let box = result[0].dataSync()
                    let landmarks = result[1].arraySync()
                    ctx.beginPath()
                    ctx.rect(box[0], box[1], box[2]-box[0], box[3]-box[1])
                    ctx.stroke()
                    for (let point of landmarks){
                        ctx.beginPath()
                        ctx.rect(point[0], point[1], 5, 5)
                        ctx.stroke()
                    }
                }
            }
        )
        console.log("Is this blocked?")
        waitingTxt.style.visibility = "visible"
    }
    image.src = URL.createObjectURL(imageInput.files[0]);
}