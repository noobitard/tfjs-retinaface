// let net
// (async function initNet(){
//     net = await tf.loadLayersModel('/exported/model.json')
//     console.log('Successfully loaded model')
// })()

function getFakeDomElement(node) {
    const fakeNode = {
        nodeName: node.nodeName,
        nodeType: node.nodeType,
        tagName: node.tagName,
        childNodes: [...node.childNodes].map(child => getFakeDomElement(child)),
        textContent: node.textContent
    }
    if(node.attributes) {
        fakeNode.attributes = [...node.attributes].map(attribute => ({name:attribute.name, value:attribute.value}))
    }
    return fakeNode;
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
        imagedata = ctx.getImageData(0, 0, resizeWidth, resizeHeight)

        let worker = new Worker("worker.js")
        worker.postMessage({"image": imagedata, "height": resizeHeight, "width": resizeWidth})
        worker.onmessage = (message) => {
            results = message.data["results"]
            waitingTxt.style.visibility = "hidden"
            for (let result of results[0]){
                let box = result[0]
                let landmarks = result[1]
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
        waitingTxt.style.visibility = "visible"
    }
    image.src = URL.createObjectURL(imageInput.files[0]);
}