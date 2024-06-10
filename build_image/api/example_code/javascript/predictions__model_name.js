const imageData = fs.readFileSync('cocoa.jpg');

// Get the binary model prediction for image cocoa.jpg 
// passed as a binary file in the request body
fetch.then(async fetch => {
    const response_binary = await fetch(
        "https://api-test.openepi.io/crop-health/predictions/binary",
        {
            method: "POST",
            body: imageData,
        }
    );
    const data_binary = await response_binary.json();
    // Print the prediction for the healthy class
    console.log(data_binary.HLT);
});


// Get the single-HLT model prediction for image cocoa.jpg 
// passed as a binary file in the request body
fetch.then(async fetch => {
  const response_single = await fetch(
      "https://api-test.openepi.io/crop-health/predictions/single-HLT",
      {
          method: "POST",
          body: imageData,
      }
  );
  const data_single = await response_single.json();
  // Print the top 5 predictions
  console.log(data_single);
});

// Get the multi-HLT prediction for image cocoa.jpg 
// passed as a binary file in the request body
fetch.then(async fetch => {
  const response_multi = await fetch(
      "https://api-test.openepi.io/crop-health/predictions/multi-HLT",
      {
          method: "POST",
          body: imageData,
      }
  );
  const data_multi = await response_multi.json();
  // Print the top 5 predictions
  console.log(data_multi);
});
