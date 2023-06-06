name=aii_admin_backend
tag=chat_backend_test7
container_name=aii_chat_backend
run:
	docker run --network host --name $(container_name) -v /etc/aii/$(container_name).env:/app/.env -d  $(name):$(tag)
run_prod:
	docker run --network host --name aii_chat_backend --env-file /etc/aii/aii_chat_backend.env -v /etc/aii/aii_chat_backend.env:/app/.env -v /var/log/aii/:/app/log -d  dextr/aii_admin_backend:chat_backend_test7
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