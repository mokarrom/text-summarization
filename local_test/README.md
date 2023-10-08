
# Running and Testing Docker Image Locally

- Build a docker image
> _docker build --no-cache -f Dockerfile -t chapt_sum ._
- Enter the `local_test` folder
> _cd local_test_
- Run the container
> _./serve_local.sh chapt_sum_
- Execute payload in another tab 
> _./summarize.sh payload.json_
