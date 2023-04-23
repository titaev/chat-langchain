name=aii_admin_backend
tag=chat_backend_test1
container_name=aii_chat_backend
run:
	docker run --network host --name $(container_name) -v /etc/aii/$(container_name).env:/app/.env -d  $(name):$(tag)
run_prod:
	docker run --restart always --network host --name $(container_name) -v /etc/aii/$(container_name).env:/app/.env -d  $(name):$(tag)
build:
	docker build -t $(name):$(tag) .
stop:
	docker stop $(container_name)
rm:
	docker rm $(container_name)
push:
	docker tag $(name):$(tag) dextr/$(name):$(tag) && docker push dextr/$(name):$(tag)
update:
	make build && make push
rerun:
	make build && make stop && make rm && make run