const image = new FormData();
image.append('file', fs.createReadStream('cocoa.jpg'));

const response_binary = await fetch(
    "$api_url" + "/predictions/binary", {
      method: "POST",
      body: image
    }
  );
const data_binary = await response_binary.json();
  
console.log(data_binary.HLT);

const response_single_HLT = await fetch(
    "$api_url" + "/predictions/single-HLT", {
      method: "POST",
      body: image
    }
  );
const data_single_HLT = await response_single_HLT.json();

console.log(data_single_HLT);

const response_multi_HLT = await fetch(
      "$api_url" + "/predictions/multi-HLT", {
        method: "POST",
        body: image
      }
    );
const data_multi_HLT = await response_multi_HLT.json();
  
console.log(data_multi_HLT);
